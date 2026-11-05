"""Recover discharge, or explicitly reuse unchanged historical DOE observations.

Never synthesize a raw repose/rolling measurement: their original CSV payloads
remain byte-for-byte field values in the merged observation. A technical failure
is a terminal *attempt*, not a 90-degree physical response. The DOE row remains
complete as an imported observation container, with discharge explicitly missing.
One reserved slot is consumed even on failure; orphan reservations fail closed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath

from optimizer.calibration_controller import (
    CalibrationController, RESULT_FIELDS, TESTS, _trial_json_ready,
    is_valid_liggghts_seed,
)
from optimizer.adaptive_outcomes import interpret_response, strict_bool
from optimizer.adaptive_refinement import _lock


def _atomic_json(path, payload):
    temporary = path.with_name('.'+path.name+'.recovery.tmp')
    with temporary.open('w', encoding='utf8') as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _journal(path, event, **fields):
    with path.open('a', encoding='utf8') as handle:
        handle.write(json.dumps(dict(event=event, **fields), allow_nan=False, sort_keys=True)+'\n')
        handle.flush()
        os.fsync(handle.fileno())

DOE_COUNT = 252
NEW_BUDGET = 748
MAX_WORKERS = 12
KINDS = ('discharge', 'repose', 'rolling')
FACTORS = {'sf11', 'sf12', 'rf11', 'rf12', 'CED11', 'CED12'}
HISTORICAL_REUSE_POLICY = 'reuse-historical-responses-new-trials-only'


def reuse_historical_responses(controller):
    """Only this exact opt-in disables recovery; legacy behavior is unchanged."""
    return (controller.config.get('sequential_experiment', {}).get('recovery_policy') ==
            HISTORICAL_REUSE_POLICY)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize_source_row(source):
    """Preserve compatible fields, erase every historical discharge datum."""
    row = dict(source)
    for key in tuple(row):
        if key.startswith('discharge_'):
            row[key] = ''
    row.update(discharge_deg='', discharge_raw_deg='', discharge_raw_status='missing',
               discharge_response_status='missing', discharge_measurement_valid=False,
               discharge_raw_reason='new_mass_protocol_recovery_pending',
               discharge_assignment_reason='no_new_discharge_measurement',
               discharge_physical_outcome_verified=False,
               discharge_physical_outcome_evidence='', score='', accepted=False,
               status='complete', phase='doe', workspace='')
    return row


def _read_csv(path):
    if not Path(path).exists():
        return []
    # Actual historical rolling QC JSON exceeds Python's default 128 KiB limit.
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _static_map(path):
    data = json.loads(path.read_text(encoding='utf8'))
    return {key.replace('\\', '/'): value for key, value in data.get('files', data).items()}


def _safe_file(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in relative or '\\' in relative:
        raise ValueError('unsafe manifest path')
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest path outside isolated project')
    return path


def _verify_future_repose_overlay(controller, relative, baseline_digest):
    """Permit one audited future-only charge overlay, not protocol relabeling.

    The original manifest/config remain immutable statements about imported
    responses. The sidecar records the subsequent template and trial boundary.
    """
    root = controller.project_root
    sidecar = root/'runtime/future_repose_overlay.json'
    if relative != 'Ang Repouso/in.repouso' or not sidecar.exists():
        return False
    meta = json.loads(sidecar.read_text(encoding='utf8'))
    if meta.get('schema_version') == 2:
        from optimizer.repose_ansys import verify_overlay
        if not reuse_historical_responses(controller) or meta.get('baseline_sha256') != baseline_digest:
            raise ValueError('ANSYS future overlay baseline/policy mismatch')
        return verify_overlay(root)
    if (not reuse_historical_responses(controller) or
            meta.get('schema_version') != 1 or meta.get('assay') != 'repose' or
            meta.get('relative_input') != relative or
            meta.get('mass_multiplier') != 1.25 or
            type(meta.get('first_trial_id')) is not int or
            not DOE_COUNT < meta['first_trial_id'] <= 1000 or
            meta.get('preserve_prior_trials') is not True or
            meta.get('historical_reuse_policy') != HISTORICAL_REUSE_POLICY or
            meta.get('baseline_sha256') != baseline_digest):
        raise ValueError('invalid future repose charge overlay scope')
    before = _safe_file(root, meta['baseline_relative_path'])
    current = _safe_file(root, relative)
    if _sha(before) != baseline_digest or _sha(current) != meta.get('revised_sha256'):
        raise ValueError('future repose charge overlay hash mismatch')
    original = before.read_bytes()
    old = b'variable charge_multiplier equal 1.0'
    new = b'variable charge_multiplier equal 1.25'
    if original.count(old) != 1:
        raise ValueError('future repose baseline must contain one charge multiplier')
    expected = original.replace(old, new, 1)
    box15 = meta.get('protocol_revision') == 'rectangular-base-clearance1mm-total15s'
    final_time, settle, steps = (15.0, 2.0, 300000) if box15 else (13.0, 0.0, 260000)
    if (meta.get('final_time_s') != final_time or meta.get('post_tube_settle_s') != settle
            or meta.get('timestep_s') != 0.00005 or meta.get('total_steps') != steps
            or 'timing_first_trial_id' in meta):
        raise ValueError('invalid future repose timing scope')
    if box15:
        if (meta.get('box_timing_first_trial_id') != 556 or
                meta.get('launch_authorized') is not False or
                meta.get('clearance_original_units') != 0.001 or
                meta.get('box_original_bounds') != [-0.301, 0.301, -0.001, 2.5, -0.301, 0.301] or
                meta.get('previous_overlay_sha256') != '604685718460ed1fb0d2242b8b4f2e45a1c99ea8fc8ae5a41ee86634f13abb27' or
                _sha(_safe_file(root, meta['previous_overlay_relative_path'])) != meta['previous_overlay_sha256']):
            raise ValueError('invalid future 15s rectangular box scope/provenance')
    # Exact reviewed transformations of the immutable baseline; reject all other
    # edits, including any extra run or post-tube settling, even if rehashed.
    nl = b'\r\n' if b'\r\n' in expected else b'\n'
    replacements = [
        (b'variable dt equal 1e-4', b'variable dt equal 5e-5'),
        (b'thermo 1000 # grava o progresso a cada 0,1 s' + nl
         + b'thermo_style custom step time atoms ke # envia Step, Time, Atoms e energia cin\xc3\xa9tica ao log' + nl
         + nl + b'variable dumptime equal 0.1' + nl
         + b'variable dumpstep equal ${dumptime}/${dt} # n\xc3\xbamero de passos entre arquivos de sa\xc3\xadda',
         b'variable dumptime equal 0.1' + nl
         + b'variable dumpstep equal round(v_dumptime/v_dt)' + nl
         + b'thermo ${dumpstep} # grava o progresso a cada 0,1 s' + nl
         + b'thermo_style custom step time atoms ke # envia Step, Time, Atoms e energia cin\xc3\xa9tica ao log' + nl
         + b'thermo_modify flush yes # libera cada amostra para o log sem alterar o intervalo f\xc3\xadsico' + nl),
    ]
    for old, new in replacements:
        if expected.count(old) != 1:
            raise ValueError('unexpected repose baseline for reviewed timestep/cadence overlay')
        expected = expected.replace(old, new, 1)
    if box15:
        # Apply only the authorized bounds/time delta to the reviewed 13 s deck.
        delta = [
            (b'variable    xmin    index    -1.0 ', b'variable    xmin    index    -0.301 '),
            (b'variable    xmax    index     1.0 ', b'variable    xmax    index     0.301 '),
            (b'variable    ymin    index    -0.5 ', b'variable    ymin    index    -0.001 '),
            (b'variable    zmin    index    -1. ', b'variable    zmin    index    -0.301 '),
            (b'variable    zmax    index     1. ', b'variable    zmax    index     0.301 '),
            (b'variable simulationtime equal 13.0 ', b'variable simulationtime equal 15.0 '),
            (b'write_dump all custom/vtk post/repouso_10s.vtk',
             b'variable repose_settle_steps equal round(2.0/${dt})' + nl
             + b'run ${repose_settle_steps} # settle 13 to 15 s; tube stationary' + nl
             + b'write_dump all custom/vtk post/repouso_10s.vtk'),
        ]
        for old, new in delta:
            if expected.count(old) != 1:
                raise ValueError('unexpected repose baseline for 15s box overlay')
            expected = expected.replace(old, new, 1)
    if current.read_bytes() != expected:
        raise ValueError('future repose overlay differs from exact reviewed charge/timestep/cadence/box/timing scope')
    return True


def _verify_historical_row(row, original, source_hash):
    """Historical fields retain their source meaning, including old workspace.

    Compare strings exactly, not numerical approximations or recovery artifacts.
    Optional provenance may deny, never assert, new-mass validation. Empty extra
    columns are harmless when adaptive rows extend the common CSV schema.
    """
    for key, value in original.items():
        if row.get(key) != value:
            raise ValueError('historical field changed: '+str(key))
    provenance = {
        'historical_reuse_policy': HISTORICAL_REUSE_POLICY,
        'historical_source_doe_sha256': source_hash,
        'historical_new_mass_validated': 'False',
    }
    for key, value in row.items():
        if key not in original and value not in ('', None):
            if key not in provenance or value != provenance[key]:
                raise ValueError('invalid historical provenance: '+str(key))


# Explicit, immutable reviewed future-deck exceptions. Historical manifest is not rewritten.
_REVIEWED_FUTURE_DECKS = {'Ang Deslizamento/in.deslizamento': ('edb477fc2df81f5b9750ce32d567799fec5c4bca0614f71ffe412f23d7c1ffa7', 'c792eaa4bc319152a9390841246324afe33674104be7536c47d748f44a4c85f3'), 'Ang Rolamento/in.fill_1s': ('c418a4ce397ba2fa48472e9484e201509a140d21bcfc2af201e8b509cf0b8ebb', 'd51cd4ffa062c2de082245242b4edd40035312e61f8168bf82f635e4c95d7534'), 'Ang Rolamento/in.rolamento': ('8196a622795361f58f98d63653e9afc43e1d6b9e78d222b29060e68e30d37eb9', '1202819913ee39b7a642eaf1b2ca93f03aa464ff49df3ece2c66a53bee57e12d')}

def _verify_reviewed_future_deck(controller, relative, baseline_digest):
    pin = _REVIEWED_FUTURE_DECKS.get(relative)
    if not pin or not reuse_historical_responses(controller):
        return False
    if baseline_digest != pin[0]:
        raise ValueError('reviewed future deck baseline mismatch: '+relative)
    actual = _sha(_safe_file(controller.project_root, relative))
    if actual == pin[1]:
        return True
    from optimizer.transition_guards import CANDIDATE_SHA256, verify_candidate_pins
    if actual != CANDIDATE_SHA256.get(relative):
        raise ValueError('reviewed future deck provenance mismatch: '+relative)
    verify_candidate_pins(controller.project_root)
    return True


def _source_identity_matches(key, source, current, authorized_transition=False):
    if source == current:
        return True
    if key != 'rolling_protocol' or not authorized_transition or not isinstance(source, dict) or not isinstance(current, dict):
        return False
    old, new = dict(source), dict(current)
    old_dt, new_dt = old.pop('dt_s', None), new.pop('dt_s', None)
    return old == new and old_dt in (None, 0.00005) and new_dt == 0.0001


def verify_sequential_manifest(controller):
    """Offline integrity checks, including source identity and reused protocols.

    The import manifest is authoritative; sequential_manifest is accepted for
    older prepared bundles with the same semantics. No source campaign is read.
    """
    root = controller.project_root
    path = root/'runtime/import_manifest.json'
    if not path.exists():
        path = root/'runtime/sequential_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf8'))
    source = root/'provenance/imported_doe_results.csv'
    expected_hash = manifest.get('imported_doe_sha256', manifest.get('source_doe_sha256'))
    if _sha(source) != expected_hash:
        raise ValueError('source DOE hash mismatch')
    count = manifest.get('imported_doe_rows', manifest.get('source_doe_rows'))
    ids = manifest.get('imported_trial_ids', manifest.get('imported_doe_ids'))
    if (count != DOE_COUNT or sorted(ids or []) != list(range(1, DOE_COUNT+1)) or
            manifest.get('total_trial_slots') != 1000 or
            manifest.get('new_attempt_budget') != NEW_BUDGET or manifest.get('workers') != MAX_WORKERS):
        raise ValueError('sequential manifest budget/IDs mismatch')
    config = controller.config
    if (set(config['parameters']) != FACTORS or set(config['tests']) != set(KINDS) or
            float(config.get('charge_revision', {}).get('discharge', 0)) != 1.5 or
            int(config['campaign']['max_trials']) != 1000 or
            int(config.get('maximum_doe_attempts', DOE_COUNT)) != DOE_COUNT or
            int(config['campaign']['refinement_trials']) != NEW_BUDGET or
            int(config['campaign']['validation_trials']) != 0):
        raise ValueError('configuration does not describe the fixed 252+748 mass1.5 experiment')
    for key in ('mpi_processes', 'mpi_processes_per_trial'):
        if int(config.get(key, 1)) != 1:
            raise ValueError('recovery requires one solver per worker')
    for key in ('parallel_trials', 'max_concurrent_simulations', 'adaptive_max_inflight'):
        if not 1 <= int(config.get(key, MAX_WORKERS)) <= MAX_WORKERS:
            raise ValueError('worker ceiling exceeds twelve')
    source_identity = json.loads((root/'provenance/source_campaign_identity.json').read_text(encoding='utf8'))
    current_identity = controller._campaign_identity()
    for key in ('parameters', 'fixed_parameters', 'tests', 'rolling_protocol'):
        source_value, current_value = source_identity.get(key), current_identity.get(key)
        if source_value != current_value:
            approval_path = root/'launch_authorization.json'
            approval = json.loads(approval_path.read_text(encoding='utf8')) if approval_path.is_file() else {}
            authorized_transition = approval.get('scope') == 'authorized-sequential-dt100us-v1'
            if authorized_transition:
                from optimizer.launch_authorization import require_launch_authorization
                require_launch_authorization(root, config)
            if not _source_identity_matches(key, source_value, current_value, authorized_transition):
                raise ValueError('incompatible reused assay identity: '+key)
    if any(float(config.get('charge_revision',{}).get(kind,1.0)) != 1.0
           for kind in ('repose','rolling')):
        raise ValueError('only discharge charge may change')
    static = _static_map(root/'provenance/source_static_hashes.json')
    protocols = manifest.get('assay_protocol_sha256')
    if protocols is not None:
        if set(protocols) != set(KINDS) or protocols != config.get('assay_protocol_sha256'):
            raise ValueError('protocol manifest/config mismatch')
        for kind, files in protocols.items():
            if not files:
                raise ValueError('empty protocol hash manifest')
            for relative, digest in files.items():
                if _sha(_safe_file(root, relative)) != digest:
                    if not ((kind == 'repose' and _verify_future_repose_overlay(controller, relative, digest)) or _verify_reviewed_future_deck(controller, relative, digest)):
                        raise ValueError('assay protocol hash mismatch: '+relative)
                if kind != 'discharge' and static.get(relative) != digest:
                    raise ValueError('reused assay differs from source: '+relative)
    else:
        # Older prepared import manifests still require verified source assets.
        for relative, digest in static.items():
            if relative.startswith(('Ang Repouso/', 'Ang Rolamento/')):
                if _sha(_safe_file(root, relative)) != digest:
                    raise ValueError('reused assay hash mismatch: '+relative)
    sources = _read_csv(source)
    by_id = {}
    for row in sources:
        trial_id = int(row['trial_id'])
        if trial_id in by_id or row.get('phase') != 'doe' or row.get('status') != 'complete':
            raise ValueError('duplicate or incomplete source DOE record')
        if not controller._is_within_active_bounds(row):
            raise ValueError('invalid historical parameters')
        for name in ('distribution_seed', 'insertion_seed'):
            if not is_valid_liggghts_seed(int(row[name])):
                raise ValueError('invalid historical seed')
        by_id[trial_id] = row
    if sorted(by_id) != list(range(1, DOE_COUNT+1)):
        raise ValueError('source DOE must contain exactly IDs 1..252')
    current = _read_csv(controller.run_root/'results.csv')
    seen = set()
    for row in current:
        trial_id = int(row['trial_id'])
        if trial_id in seen:
            raise ValueError('duplicate result record')
        seen.add(trial_id)
        if trial_id > DOE_COUNT:
            if row.get('phase') == 'doe' or trial_id > 1000:
                raise ValueError('nonhistorical DOE or out-of-budget trial ID')
            continue
        if trial_id not in by_id:
            raise ValueError('unexpected historical ID')
        original = by_id[trial_id]
        if reuse_historical_responses(controller):
            _verify_historical_row(row, original, expected_hash)
            # A new-mass recovery artifact must never replace source equivalence.
            continue
        for key, value in original.items():
            if key in FACTORS or key in ('distribution_seed', 'insertion_seed') or key.startswith(('repose_', 'rolling_')):
                if str(row.get(key, '')) != value:
                    raise ValueError('historical field changed: '+key)
        if (row.get('discharge_deg') not in (None, '') or
                row.get('discharge_raw_deg') not in (None, '') or
                strict_bool(row.get('discharge_measurement_valid'))):
            artifact = controller.run_root/f'trial_{trial_id:04d}'/'trial.json'
            if not artifact.exists():
                raise ValueError('unproven discharge (possibly imported old response)')
            evidence = json.loads(artifact.read_text(encoding='utf8'))
            recovery = evidence.get('recovery', {})
            if (recovery.get('source_doe_sha256') != expected_hash or
                    recovery.get('fresh_assays') != ['discharge'] or
                    recovery.get('charge_multiplier') != 1.5 or
                    recovery.get('source_trial_id') != trial_id or
                    recovery.get('capacity_evidence',{}).get('verified') is not True or
                    str(evidence.get('discharge_deg')) != str(row['discharge_deg'])):
                raise ValueError('unproven new-mass discharge')
    return dict(manifest=manifest, source_doe_sha256=expected_hash,
                source_rows=by_id, manifest_sha256=_sha(path))


def _write_results(controller, rows):
    """Atomic durable replacement; caller holds controller's CSV lock."""
    path = controller.run_root/'results.csv'
    temporary = path.with_name('.results.recovery.tmp')
    fields = list(dict.fromkeys(list(RESULT_FIELDS) + [key for row in rows for key in row]))
    with temporary.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    controller._results_revision += 1


def initialize_sanitized_results(controller, verified=None):
    """Idempotently import the sanitized DOE; never repair dirty legacy data."""
    if reuse_historical_responses(controller):
        raise ValueError('historical sanitization is disabled by explicit reuse policy')
    verified = verified or verify_sequential_manifest(controller)
    with controller._csv_lock:
        rows = _read_csv(controller.run_root/'results.csv')
        seen = {int(row['trial_id']) for row in rows}
        missing = [sanitize_source_row(row) for trial_id, row in sorted(verified['source_rows'].items())
                   if trial_id not in seen]
        if missing:
            _write_results(controller, rows + missing)
    return len(missing)


def initialize_historical_results(controller, verified=None):
    """Import all source fields verbatim; never sanitize/repair existing rows.

    Caller holds the adaptive lock. Refresh verification under the CSV lock so
    a stale snapshot cannot hide partially reassayed or edited historical rows.
    """
    if not reuse_historical_responses(controller):
        raise ValueError('historical initialization requires the exact explicit opt-in policy')
    with controller._csv_lock:
        verified = verify_sequential_manifest(controller)
        rows = _read_csv(controller.run_root/'results.csv')
        seen = {int(row['trial_id']) for row in rows}
        missing = [dict(row) for trial_id, row in sorted(verified['source_rows'].items())
                   if trial_id not in seen]
        if missing:
            _write_results(controller, sorted(rows + missing, key=lambda row: int(row['trial_id'])))
    return len(missing)


class DischargeCapacityError(RuntimeError):
    """The new requested/inserted charge was not demonstrated by the solver log."""


def discharge_capacity_evidence(controller, result):
    """Check the expanded LIGGGHTS request AND cumulative actually inserted mass.

    Immutable deck constants imply old mass=957000 and new mass=1435500 in
    this protocol's simulation units. A valid angle alone is NOT capacity proof.
    Allow 1% insertion discreteness, not the old charge's 33% shortfall.
    """
    import re
    trial = controller.run_root/f"trial_{int(result['trial_id']):04d}"
    # LIGGGHTS echoes expanded input to its own log, not necessarily screen.
    # Keep captured-screen fallback for historical echo-both logs.
    solver_log = trial/'Ang Deslizamento'/'log.liggghts'
    log = solver_log if solver_log.is_file() else trial/'discharge.log'
    evidence = dict(verified=False, expected_requested_mass=1435500.0,
                    relative_mass_tolerance=0.01, log_path=str(log))
    if not log.is_file():
        return dict(evidence, reason='missing_discharge_log')
    text=log.read_text(encoding='utf8', errors='replace')
    number=r'([+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'
    requests=re.findall(r'^\s*fix\s+ins\s+all\s+insert/stream\b[^\n]*?\bmass\s+'+number+r'\b',text,re.MULTILINE)
    totals=re.findall(r'a total of\s+\d+\s+particle templates?\s*\(mass\s+'+number+r'\)',text,re.IGNORECASE)
    evidence['log_sha256']=_sha(log)
    if not requests or not totals:
        return dict(evidence, reason='missing_expanded_request_or_inserted_mass')
    requested=float(requests[-1]); inserted=float(totals[-1])
    evidence.update(requested_mass=requested, inserted_mass=inserted,
                    inserted_fraction=inserted/requested if requested>0 else None)
    if not math.isclose(requested,1435500.0,rel_tol=1e-6):
        return dict(evidence, reason='requested_mass_is_not_the_new_1_5_protocol')
    if not math.isclose(inserted,requested,rel_tol=0.01):
        return dict(evidence, reason='inserted_mass_outside_one_percent_of_new_request')
    return dict(evidence, verified=True, reason='requested_and_inserted_new_charge_verified')


class DischargeRecoveryController(CalibrationController):
    """Dedicated instance, full physical identity, discharge-only execution view."""
    def __init__(self, project_root, run_root=None, **kwargs):
        super().__init__(project_root, run_root or Path(project_root)/'run', **kwargs)
        if reuse_historical_responses(self):
            raise ValueError('historical discharge recovery is disabled by explicit reuse policy')
        self.verified = verify_sequential_manifest(self)
        self.source_rows = self.verified['source_rows']
        initialize_sanitized_results(self, self.verified)

    @property
    def tests(self):
        return tuple(test for test in CalibrationController.tests.fget(self) if test[0] == 'discharge')

    @property
    def response_scales(self):
        return {kind: self.config.get('response_scales', {}).get(kind, self.config.get('tolerance_deg', .5))
                for kind in KINDS}

    def _valid_responses(self, row):
        from optimizer.protocol_eligibility import interpret_for_controller
        return all(interpret_for_controller(self, row, kind)['regression_included'] for kind in KINDS)

    def responses_accepted(self, row):
        return self._valid_responses(row) and all(
            row.get(kind+'_raw_status') == 'measured' and
            strict_bool(row.get(kind+'_measurement_valid')) is True and
            abs(float(row[kind+'_deg'])-float(self.config['target_'+kind+'_deg'])) <= float(self.response_scales[kind])+1e-9
            for kind in KINDS)

    def _validate_source_call(self, trial_id, parameters, seeds):
        row = self.source_rows[int(trial_id)]
        if set(parameters) != set(self.parameter_names) or any(
                float(parameters[key]) != float(row[key]) for key in self.parameter_names):
            raise ValueError('recovery parameters differ from source')
        original = {key: int(row[key+'_seed']) for key in ('distribution', 'insertion')}
        if seeds is not None and seeds != original:
            raise ValueError('recovery seeds differ from source')
        return original

    def prepare_trial(self, trial_id, parameters, seeds=None):
        return super().prepare_trial(trial_id, parameters,
                                     self._validate_source_call(trial_id, parameters, seeds))

    def execute_trial(self, trial_id, parameters, phase='doe', seeds=None):
        if phase != 'doe':
            raise ValueError('recovery only accepts historical DOE IDs')
        original_seeds = self._validate_source_call(trial_id, parameters, seeds)
        return super().execute_trial(trial_id, parameters, phase='doe', seeds=original_seeds)

    def _merge(self, result):
        original = self.source_rows[int(result['trial_id'])]
        merged = sanitize_source_row(original)
        for key, value in result.items():
            if key.startswith('discharge_') or key in ('workspace', 'measurement_method', 'response_policy'):
                merged[key] = value
        merged['accepted'] = self.responses_accepted(merged)
        merged['score'] = ''  # full controller scores all assays after recovery drains
        return merged

    def _persist(self, merged, failure=None):
        trial_id = int(merged['trial_id'])
        workspace = self.run_root/f'trial_{trial_id:04d}'
        workspace.mkdir(exist_ok=True)
        artifact_path = workspace/'trial.json'
        artifact = json.loads(artifact_path.read_text(encoding='utf8')) if artifact_path.exists() else {}
        responses = artifact.get('responses', {})
        reused = {kind: {key: value for key, value in self.source_rows[trial_id].items()
                         if key.startswith(kind+'_')} for kind in ('repose', 'rolling')}
        artifact.update(merged)
        artifact.update(responses=responses, recovery={
            'source_doe_sha256': self.verified['source_doe_sha256'],
            'source_trial_id': trial_id, 'charge_multiplier': 1.5,
            'fresh_assays': ['discharge'], 'reused_assays': reused,
            'attempt_status': 'failed' if failure else 'completed',
            'capacity_evidence': discharge_capacity_evidence(self, merged),
            'error': failure, 'independent_confirmation_required': True})
        _atomic_json(artifact_path, _trial_json_ready(artifact))
        with self._csv_lock:
            rows = _read_csv(self.run_root/'results.csv')
            rows = [row for row in rows if int(row['trial_id']) != trial_id]
            rows.append(merged)
            _write_results(self, sorted(rows, key=lambda row: int(row['trial_id'])))

    def _append_results(self, result):
        capacity=discharge_capacity_evidence(self,result)
        if not capacity['verified']:
            raise DischargeCapacityError(capacity['reason'])
        merged = self._merge(result)
        self._persist(merged)
        result.clear()
        result.update(merged)  # base execute_trial returns the merged observation

    def record_failure(self, trial_id, error):
        merged = sanitize_source_row(self.source_rows[int(trial_id)])
        merged.update(discharge_raw_status='technical_failure',
                      discharge_response_status='technical_failure',
                      discharge_raw_reason=str(error),
                      discharge_assignment_reason='technical_failure_no_physical_response',
                      workspace=str(self.run_root/f'trial_{int(trial_id):04d}'))
        self._persist(merged, failure=str(error))
        return merged


def _journal_state(path, controller):
    events = []
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                raise ValueError('malformed recovery journal; manual reconciliation required') from exc
    binding = dict(source_doe_sha256=controller.verified['source_doe_sha256'],
                   config_sha256=_sha(controller.project_root/'configuracao_otimizacao.txt'),
                   campaign_identity=controller._campaign_identity(),
                   manifest_sha256=controller.verified['manifest_sha256'], budget=DOE_COUNT)
    if not events:
        _journal(path, 'recovery_started', **binding)
    elif events[0] != dict(event='recovery_started', **binding):
        raise ValueError('recovery journal binding mismatch')
    reserved, terminal = set(), {}
    for event in events[1:]:
        trial_id = int(event.get('trial_id', 0))
        if trial_id not in controller.source_rows:
            raise ValueError('out-of-budget recovery reservation')
        if event['event'] == 'reserved':
            if trial_id in reserved:
                raise ValueError('repeated recovery reservation')
            reserved.add(trial_id)
        elif event['event'] in ('completed', 'failed'):
            if trial_id not in reserved or trial_id in terminal:
                raise ValueError('invalid terminal recovery journal event')
            terminal[trial_id] = event
        else:
            raise ValueError('unknown recovery journal event')
    if reserved - terminal.keys():
        raise RuntimeError('orphan reserved recovery slots: manual reconciliation required; never automatically rerun')
    return reserved, terminal


def run_discharge_recovery(controller, *, workers=MAX_WORKERS, failure_limit=3):
    """252 one-shot slots, single-worker mass pilot, then <=12 concurrent solvers.

    Three consecutive infrastructure failures open the circuit. Already running
    jobs always drain before return. A non-measurable *physical* outcome consumes
    its slot but cannot open the capacity gate. Orphans fail closed on resume.
    """
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError('workers must be an integer between one and twelve')
    if type(failure_limit) is not int or failure_limit < 1:
        raise ValueError('failure_limit must be positive')
    from optimizer.launch_authorization import require_launch_authorization
    require_launch_authorization(controller.project_root, controller.config)
    if not controller.config.get('simulation_launch_enabled', False):
        raise RuntimeError('simulation launching disabled')
    if workers > int(controller.config.get('parallel_trials', 1)):
        raise ValueError('workers exceed configured ceiling')
    with _lock(controller.run_root):
        verify_sequential_manifest(controller)
        initialize_sanitized_results(controller, controller.verified)
        path = controller.run_root/'discharge_recovery_journal.jsonl'
        reserved, terminal = _journal_state(path, controller)
        pilot = any(event.get('capacity_pilot_passed') is True for event in terminal.values())
        consecutive_failures = 0
        for event in terminal.values():
            consecutive_failures = consecutive_failures+1 if event['event'] == 'failed' else 0
        stopped = 'infrastructure_failure_circuit' if consecutive_failures >= failure_limit else None
        if any(event.get('capacity_failure') is True for event in terminal.values()):
            stopped='capacity_pilot_failed'
        accepted = None
        for row in _read_csv(controller.run_root/'results.csv'):
            if int(row['trial_id']) in terminal and controller.responses_accepted(row):
                accepted = int(row['trial_id'])
                stopped = 'joint_candidate_requires_independent_confirmation'
                break
        remaining = iter(sorted(set(controller.source_rows)-reserved))
        exhausted = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inflight = {}
            while True:
                while not stopped and not exhausted and len(inflight) < (workers if pilot else 1):
                    trial_id = next(remaining, None)
                    if trial_id is None:
                        exhausted = True
                        break
                    source = controller.source_rows[trial_id]
                    params = {name: float(source[name]) for name in controller.parameter_names}
                    seeds = {key: int(source[key+'_seed']) for key in ('distribution','insertion')}
                    _journal(path, 'reserved', trial_id=trial_id, parameters=params, seeds=seeds)
                    reserved.add(trial_id)
                    future = pool.submit(controller.execute_trial, trial_id, params, phase='doe', seeds=seeds)
                    inflight[future] = trial_id
                if not inflight:
                    break
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    trial_id = inflight.pop(future)
                    try:
                        result = future.result()
                        capacity=discharge_capacity_evidence(controller,result)
                        if not capacity['verified']:
                            raise DischargeCapacityError(capacity['reason'])
                    except Exception as exc:
                        result = controller.record_failure(trial_id, str(exc))
                        capacity_failure=isinstance(exc,DischargeCapacityError)
                        event = dict(event='failed', trial_id=trial_id, error=str(exc), capacity_pilot_passed=False,
                                     capacity_failure=capacity_failure)
                        consecutive_failures += 1
                        if capacity_failure:
                            stopped='capacity_pilot_failed'
                        elif consecutive_failures >= failure_limit:
                            stopped = 'infrastructure_failure_circuit'
                    else:
                        passed = (result.get('discharge_raw_status') == 'measured' and
                                  strict_bool(result.get('discharge_measurement_valid')) is True and
                                  interpret_response(result, 'discharge')['regression_included'])
                        pilot = pilot or passed
                        event = dict(event='completed', trial_id=trial_id, capacity_pilot_passed=passed,
                                     accepted=controller.responses_accepted(result))
                        consecutive_failures = 0
                        if event['accepted']:
                            accepted = trial_id
                            stopped = 'joint_candidate_requires_independent_confirmation'
                    _journal(path, event['event'], **{key: value for key, value in event.items() if key != 'event'})
                    terminal[trial_id] = event
        summary = dict(status='complete' if len(terminal) == DOE_COUNT else 'stopped',
                       reserved_attempts=len(reserved), attempts=len(reserved), terminal_attempts=len(terminal),
                       failed_attempts=sum(e['event'] == 'failed' for e in terminal.values()),
                       completed=sum(e.get('error') is None for e in terminal.values()),
                       failed=sum(e.get('error') is not None for e in terminal.values()),
                       remaining=DOE_COUNT-len(reserved), budget=DOE_COUNT, workers=workers,
                       capacity_pilot_passed=pilot, stop_reason=stopped,
                       circuit_open=stopped == 'infrastructure_failure_circuit', workers_drained=True,
                       accepted_trial_id=accepted, independent_confirmation_required=True)
        _atomic_json(controller.run_root/'discharge_recovery_summary.json', summary)
        return summary
