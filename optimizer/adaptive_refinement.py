"""Refinement-only orchestration. Importing this module never launches anything."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from optimizer.article_optimizer import weighted_normalized_score
from optimizer.adaptive_outcomes import VERSION as OUTCOME_VERSION, interpret_response, write_outcomes
from optimizer.protocol_eligibility import interpret_for_controller
from optimizer.calibration_controller import RESULT_FIELDS, refresh_live_state, render_dashboard, write_live_status_csv


def score_rows(controller, rows):
    """Copy/score adopted responses in memory, preserving raw data and flags."""
    scored = []
    for raw in rows:
        r = dict(raw)
        try:
            outcomes = {k: interpret_for_controller(controller, r, k, controller.config['target_'+k+'_deg'])
                        for k in controller.config['tests']}
            errors = {k: o['error_deg'] for k,o in outcomes.items() if o['regression_included']}
            r['score'] = (weighted_normalized_score(errors, controller.response_scales, controller.config['weights'])
                          if len(errors) == len(outcomes) else None)
            r.update({k+'_error': v for k, v in errors.items()})
        except (KeyError, TypeError, ValueError):
            r['score'] = None
        r['accepted'] = _accepted(controller, r)
        scored.append(r)
    return scored


def _accepted(c, row):
    try:
        return row.get('status') == 'complete' and all(
            interpret_for_controller(c, row, k)['regression_included']
            and abs(float(row[k+'_deg'])-c.config['target_'+k+'_deg']) <= c.config['tolerance_deg']+1e-9
            for k in c.config['tests'])
    except (KeyError, TypeError, ValueError):
        return False


def _append(path, event):
    encoded = json.dumps(event, allow_nan=False, sort_keys=True)+'\n'
    with path.open('a', encoding='utf8') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _lock(root):
    import fcntl  # Linux/WSL deployment; lock file deliberately remains after exit.
    import stat
    path = root/'.adaptive_refinement.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
        os.close(fd)
        raise ValueError('unsafe lock file')
    with os.fdopen(fd, 'a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('adaptive controller lock already active') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def _point(c, point, exact=False):
    if exact and set(point) != set(c.parameter_names):
        raise ValueError('proposal requires exactly configured parameters')
    try:
        p = {n: float(point[n]) for n in c.parameter_names}
        if not c._is_within_active_bounds(p):
            raise ValueError('invalid parameters outside bounds')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('finite parameters within bounds required') from exc
    return p


def _distance(c, point, attempts):
    return min((math.sqrt(sum(((point[n]-float(a[n]))/(s['max']-s['min']))**2
                            for n, s in c.config['parameters'].items())) for a in attempts), default=math.inf)


def _configured_budget_limit(controller, imported_count=None, *, doe_records=None):
    """Hard campaign capacity, separate from the legacy 48-attempt initial default."""
    config = controller.config
    campaign = config['campaign']
    refinement = campaign['refinement_trials']
    if type(refinement) is not int or refinement < 0:
        raise ValueError('configured refinement budget must be a nonnegative integer')
    if config.get('doe_attempt_multiplier') == 5:
        # Only the owning DOE scheduler supplies live reservations. Offline
        # callers retain strict orphan reconciliation through the normal restore.
        records = controller._restore_async_doe_state() if doe_records is None else doe_records
        nonrefinement = sum(r['phase'] != 'refinement' for r in records.values())
        available = campaign['max_trials'] - nonrefinement
        if available < 0:
            raise ValueError('persisted attempts exceed campaign capacity')
        return min(refinement, available)
    scope = config.get('sequential_experiment')
    imported = campaign['doe_trials'] if imported_count is None else imported_count
    limits = [refinement]
    if 'sequential_experiment' in config:
        if (not isinstance(scope, dict)
                or any(type(scope.get(k)) is not int or scope[k] < 0
                       for k in ('total_trial_slots', 'imported_doe_rows', 'new_attempt_budget'))
                or type(scope.get('workers', 12)) is not int or not 1 <= scope.get('workers', 12) <= 12):
            raise ValueError('invalid explicit sequential experiment budget/workers')
        imported = scope['imported_doe_rows']
        if imported_count is not None and imported_count != imported:
            raise ValueError('sequential imported DOE row count differs from configuration')
        limits.extend([scope['new_attempt_budget'], scope['total_trial_slots']-imported])
    maximum = campaign.get('max_trials', refinement+imported)
    if type(maximum) is not int or maximum < 0 or type(imported) is not int or imported < 0:
        raise ValueError('campaign slots and imported DOE count must be nonnegative integers')
    limits.append(maximum-imported)
    if min(limits) < 0:
        raise ValueError('imported DOE rows exceed campaign capacity')
    return min(limits)


def _settings(c):
    settings = dict(identity=c._campaign_identity(), targets={k: c.config['target_'+k+'_deg'] for k in c.config['tests']},
                outcome_interpretation_version=OUTCOME_VERSION,
                scales=c.response_scales, tolerance_deg=c.config['tolerance_deg'],
                weights=c.config['weights'], measurement=c.config.get('measurement'),
                random_seed=c.config.get('random_seed', 1), pool=c.config.get('model_candidate_pool', 6000),
                mc_samples=c.config.get('adaptive_mc_samples', 128), min_distance=c.config.get('adaptive_min_distance', .01))
    # Omit the absent policy to preserve existing journals and their worker flexibility.
    # Enabling/changing an explicit cap requires a matching journal, never migration.
    if 'sequential_experiment' in c.config:
        from optimizer.sequential_surrogate import VERSION as SEQUENTIAL_VERSION
        settings['sequential_experiment'] = c.config['sequential_experiment']
        settings['sequential_proposal_version'] = SEQUENTIAL_VERSION
        settings['sequential_proposal_options'] = c.config.get('adaptive_refinement', {})
    if 'adaptive_doe_handoff' in c.config:
        if type(c.config['adaptive_doe_handoff']) is not bool:
            raise ValueError('adaptive_doe_handoff must be a boolean')
        settings['adaptive_doe_handoff'] = c.config['adaptive_doe_handoff']
    if 'adaptive_initial_inflight' in c.config:
        settings['adaptive_initial_inflight'] = c.config['adaptive_initial_inflight']
    return settings


def _restore(c, budget, *, allow_inflight=False, doe_records=None):
    root = c.run_root
    rows = score_rows(c, c._load_results_csv())
    ids = [int(r['trial_id']) for r in rows]
    if len(set(ids)) != len(ids) or any(i < 1 for i in ids):
        raise ValueError('duplicate/invalid result IDs')
    doe = [r for r in rows if r.get('phase') == 'doe']
    if budget > _configured_budget_limit(c, imported_count=len(doe), doe_records=doe_records):
        raise ValueError('attempt budget exceeds configured campaign capacity')
    # A completed physical attempt may have no measurable angle. The proposer
    # validates measured training support; missing angles never require imputation.
    if c.config.get('doe_attempt_multiplier') != 5 and any(r.get('status') != 'complete' for r in doe):
        raise ValueError('complete DOE records required; no new DOE allowed')
    if c.config.get('adaptive_doe_handoff') is True or c.config.get('doe_attempt_multiplier') == 5:
        # Readiness uses persisted physical attempts and measured-data rank.
        # Missing responses are never converted into fictitious angles.
        if not c._doe_is_ready(rows):
            raise ValueError('DOE is not ready for adaptive refinement')
    elif ('sequential_experiment' not in c.config
          and len(doe) != int(c.config['campaign']['doe_trials'])):
        raise ValueError('exact configured complete DOE count required; no new DOE allowed')
    if any(r.get('phase') not in ('doe', 'refinement') for r in rows):
        raise ValueError('only DOE plus journaled adaptive refinement allowed')
    for r in rows:
        _point(c, r)
    journal = root/'adaptive_proposals.jsonl'
    events = [json.loads(line) for line in journal.read_text(encoding='utf8').splitlines()] if journal.exists() else []
    reserved, terminal = {}, {}
    if journal.exists() and (not events or events[0].get('event') != 'initialized' or events[0].get('version') != 1
                   or sum(e.get('event') == 'initialized' for e in events) != 1):
        raise ValueError('missing/invalid journal initialization')
    effective_budget = None
    for e in events:
        kind = e['event']
        if kind == 'initialized':
            initial_limit = _configured_budget_limit(c, imported_count=len(doe), doe_records=doe_records)
            if 'sequential_experiment' not in c.config:
                initial_limit = min(48, initial_limit)
            if type(e.get('budget')) is not int or not 0 <= e['budget'] <= initial_limit:
                raise ValueError('invalid initial journal budget')
            effective_budget = e['budget']
            if e.get('configuration') != _settings(c):
                raise ValueError('journal scoring/model configuration differs on resume')
        elif kind == 'budget_extended':
            if (type(e.get('previous_budget')) is not int or e['previous_budget'] != effective_budget
                    or type(e.get('additional_attempts')) is not int or not 1 <= e['additional_attempts'] <= 48
                    or type(e.get('budget')) is not int or e['budget'] != effective_budget + e['additional_attempts']
                    or reserved.keys()-terminal.keys()
                    or (len(reserved) != effective_budget and e.get('allow_unspent') is not True)
                    or ('allow_unspent' in e and type(e['allow_unspent']) is not bool)):
                raise ValueError('invalid journal budget extension')
            effective_budget = e['budget']
        elif kind == 'reserved':
            if effective_budget is None or len(reserved) >= effective_budget:
                raise ValueError('journal reservations exceed authorized budget')
            if type(e.get('trial_id')) is not int or e['trial_id'] < 1:
                raise ValueError('invalid journal trial ID')
            tid = e['trial_id']
            if tid in reserved or tid in ids and next(r for r in rows if int(r['trial_id']) == tid)['phase'] != 'refinement':
                raise ValueError('duplicate/conflicting journal reservation')
            _point(c, e['parameters'], exact=True)
            reserved[tid] = e
        elif kind == 'abandoned_batch':
            batch = e.get('trial_ids')
            if (not isinstance(e.get('transaction_id'), str) or not e['transaction_id']
                    or not isinstance(batch, list) or not batch
                    or any(type(t) is not int or t < 1 for t in batch)
                    or batch != sorted(set(batch))
                    or any(t not in reserved or t in terminal or t in ids for t in batch)):
                raise ValueError('invalid abandoned batch')
            for tid in batch:
                terminal[tid] = dict(event='abandoned', trial_id=tid, transaction_id=e['transaction_id'])
        elif kind in ('completed', 'failed'):
            if type(e.get('trial_id')) is not int or e['trial_id'] < 1:
                raise ValueError('invalid journal trial ID')
            tid = e['trial_id']
            if tid not in reserved or tid in terminal:
                raise ValueError('invalid terminal journal event')
            if 'result' in e:
                result = e['result']
                if (not isinstance(result, dict) or type(result.get('trial_id')) is not int
                        or result['trial_id'] != tid or result.get('phase') != 'refinement'
                        or result.get('status') != ('complete' if kind == 'completed' else 'failed')
                        or _point(c, result) != reserved[tid]['parameters']):
                    raise ValueError('terminal journal result identity/status mismatch')
            elif kind == 'completed':
                raise ValueError('completed terminal event missing result')
            terminal[tid] = e
        else:
            raise ValueError('unknown journal event')
    if events and effective_budget != budget:
        raise ValueError('resume must preserve authorized total attempt budget')
    if not allow_inflight and reserved.keys()-terminal.keys():
        raise RuntimeError('unresolved orphan reserved/running attempts; reconcile manually, never relaunch')
    if any(e.get('error_kind') == 'invalid_result' for e in terminal.values()):
        raise RuntimeError('invalid result evidence requires manual reconciliation; never resume automatically')
    status = _read(root/'status.csv')
    if any(r.get('phase') == 'refinement' and int(r['trial_id']) not in reserved for r in rows+status):
        raise ValueError('legacy/untracked refinement attempt rejected')
    persisted = {int(r['trial_id']): r for r in rows}
    for tid, reservation in reserved.items():
        if tid in persisted and _point(c, persisted[tid]) != reservation['parameters']:
            raise ValueError('journal/result parameters differ')
    for tid, e in terminal.items():
        if e['event'] == 'failed' and tid in persisted and persisted[tid].get('status') != 'failed':
            raise RuntimeError('failed journal event conflicts with persisted result; reconcile manually')
        if e['event'] == 'completed' and tid not in persisted:
            raise RuntimeError('completed journal result missing from results.csv')
        if e['event'] == 'completed':
            saved, actual = e['result'], persisted[tid]
            for field in RESULT_FIELDS:
                if field in ('score', 'accepted') or field.endswith('_error'):
                    continue  # Recomputed fields are not raw measurement evidence.
                expected, observed = saved.get(field, ''), actual.get(field, '')
                expected = '' if expected is None else expected
                observed = '' if observed is None else observed
                if str(expected) == str(observed):
                    continue
                try:
                    same = math.isfinite(float(expected)) and float(expected) == float(observed)
                except (ValueError, TypeError):
                    same = False
                if not same:
                    raise ValueError('journal/result measurement mismatch: '+field)
        if tid not in persisted and e['event'] != 'abandoned':
            rows.append(dict(reserved[tid]['parameters'], trial_id=tid, phase='refinement', status='failed', score=None))
    if len(reserved) > budget:
        raise ValueError('persisted attempts exceed budget')
    used = ids+[int(r['trial_id']) for r in status]+list(reserved)
    used += [int(p.name[6:]) for p in root.glob('trial_*') if p.name[6:].isdigit()]
    attempts = [_point(c, r) for r in rows]+[e['parameters'] for e in reserved.values()]
    for path in root.glob('trial_*/parameters.json'):
        attempts.append(_point(c, json.loads(path.read_text(encoding='utf8'))['parameters'], exact=True))
    next_id = max(max(used, default=0)+1, int(c.config.get('first_trial_id', 1)), int(getattr(c, 'first_trial_id', 1)))
    return rows, reserved, terminal, attempts, next_id


def _sync_directory(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def abandon_orphans(controller, *, budget, trial_ids, transaction_id):
    """Offline only: atomically close exactly the specified entire orphan set.

    Caller must establish all workers have stopped and acquire exclusive control
    of external writers. No workspaces or results are modified. Linux/WSL only.
    """
    import tempfile
    if (not isinstance(trial_ids, list) or not trial_ids
            or any(type(t) is not int or t < 1 for t in trial_ids)
            or trial_ids != sorted(set(trial_ids))
            or not isinstance(transaction_id, str) or not transaction_id):
        raise ValueError('invalid abandonment request')
    root = Path(controller.run_root)
    journal = root/'adaptive_proposals.jsonl'
    with _lock(root):
        controller._validate_run_compatibility()
        rows, reserved, terminal, _, _ = _restore(controller, budget, allow_inflight=True)
        events = [json.loads(line) for line in journal.read_text(encoding='utf8').splitlines()]
        previous = [e for e in events if e.get('event') == 'abandoned_batch' and e.get('transaction_id') == transaction_id]
        if previous:
            if len(previous) != 1 or previous[0].get('trial_ids') != trial_ids:
                raise ValueError('transaction ID conflict')
            _sync_directory(root)  # complete a previous post-rename interrupted commit
            return
        if set(trial_ids) != reserved.keys()-terminal.keys():
            raise ValueError('request must match entire orphan set')
        if set(trial_ids) & {int(r['trial_id']) for r in rows}:
            raise ValueError('orphan conflicts with result')
        raw = journal.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('journal has incomplete final record')
        event = dict(event='abandoned_batch', trial_ids=trial_ids, transaction_id=transaction_id,
                     reason='authorized orphan abandonment', abandoned_at=time.time())
        payload = raw + (json.dumps(event, sort_keys=True, allow_nan=False)+'\n').encode('utf8')
        fd, name = tempfile.mkstemp(prefix='.abandon-', suffix='.tmp', dir=root)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, journal)
            _sync_directory(root)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def validate_adaptive_state(controller, *, budget=48):
    """Inspect adaptive recovery evidence without locks, initialization or writes."""
    rows, reserved, terminal, _, _ = _restore(controller, budget, allow_inflight=True)
    _validate_summary(controller, budget, rows, reserved, terminal)


def _validate_summary(controller, budget, rows, reserved, terminal):
    path = controller.run_root/'adaptive_summary.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf8'))
        events = [json.loads(line) for line in (controller.run_root/'adaptive_proposals.jsonl').read_text(encoding='utf8').splitlines()]
        authorized_budgets = [e['budget'] for e in events if e['event'] in ('initialized', 'budget_extended')]
        snapshot_budget = report.get('budget')
        constants = dict(repeatability_validation_performed=False,
                         physical_validation_performed=False, convergence_guaranteed=False,
                         primary_outcomes_csv='adaptive_outcomes.csv', legacy_results_csv='results.csv',
                         outcome_interpretation_version=OUTCOME_VERSION)
        if (not isinstance(report, dict) or not (controller.run_root/'adaptive_proposals.jsonl').exists()
                or type(snapshot_budget) is not int or snapshot_budget not in authorized_budgets or snapshot_budget > budget
                or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in constants.items())
                or 'configuration' in report and report['configuration'] != _settings(controller)):
            raise ValueError('adaptive summary identity/configuration mismatch')
        # A report is a derived snapshot. Counts may lag durable journal events,
        # including a completed trial whose final summary has not been replaced.
        limits = dict(attempts=len(reserved),
                      completed_attempts=sum(e['event'] == 'completed' for e in terminal.values()),
                      failed_attempts=sum(e['event'] == 'failed' for e in terminal.values()))
        candidates = report.get('candidate_trial_ids')
        accepted = {int(r['trial_id']) for r in rows if _accepted(controller, r)}
        if (any(type(report.get(k)) is not int or not 0 <= report[k] <= limit for k, limit in limits.items())
                or report['completed_attempts']+report['failed_attempts'] > report['attempts']
                or type(report.get('workers')) is not int
                or not 1 <= report['workers'] <= min(20 if controller.config.get('fresh_finite_independent') is True else 12, int(controller.config.get('parallel_trials', 1)))
                or report.get('status') not in ('running', 'candidate_found', 'budget_exhausted', 'review_cap_reached', 'exact_12_completed', 'interrupted', 'error')
                or report.get('status') == 'exact_12_completed' and (report.get('exact_12_new') is not True or report.get('invocation_target') != report['attempts'] or len(terminal) != len(reserved))
                or report.get('status') == 'review_cap_reached' and (len(reserved) < 303 or len(terminal) != len(reserved) or bool(accepted))
                or not isinstance(candidates, list) or any(type(tid) is not int for tid in candidates)
                or len(set(candidates)) != len(candidates) or not set(candidates) <= accepted
                or type(report.get('candidate_found')) is not bool or report['candidate_found'] != bool(candidates)
                or report['status'] == 'candidate_found' and not candidates
                or report['status'] == 'budget_exhausted' and (
                    report['attempts'] != snapshot_budget or accepted
                    or report['completed_attempts']+report['failed_attempts'] != snapshot_budget)):
            raise ValueError('adaptive summary differs from journal/results')


def _save_summary(root, summary):
    temporary = root/'.adaptive_summary.tmp'
    with temporary.open('w', encoding='utf8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(root/'adaptive_summary.json')


def _finish_scores(c):
    # Workers have drained. Score without touching canonical measurement evidence
    # until the single atomic replacement below.
    rows = score_rows(c, c._load_results_csv())
    path = c.run_root/'.adaptive_results.tmp'
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(c._result_csv_row(row) for row in rows)
        handle.flush()
        os.fsync(handle.fileno())
    from optimizer.protocol_eligibility import overlay_config
    schema2=(overlay_config(c.project_root) or {}).get('schema_version') == 2
    path.replace(c.run_root/('results_recalculated.csv' if schema2 else 'results.csv'))
    directory = os.open(c.run_root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def authorize_budget_extension(controller, *, current_budget, additional_attempts, allow_unspent=False):
    """Explicitly authorize more attempts on an exhausted, reconciled journal; does not launch trials."""
    if type(current_budget) is not int or current_budget < 0 or type(additional_attempts) is not int or not 1 <= additional_attempts <= 48:
        raise ValueError('extension requires an integer additional attempt budget in 1..48')
    c = controller
    with _lock(Path(c.run_root)):
        c._validate_run_compatibility()
        journal = c.run_root/'adaptive_proposals.jsonl'
        if not journal.is_file():
            raise ValueError('existing adaptive journal required')
        rows, reserved, terminal, _, _ = _restore(c, current_budget)
        _validate_summary(c, current_budget, rows, reserved, terminal)
        if type(allow_unspent) is not bool or len(terminal) != len(reserved) or (not allow_unspent and len(reserved) != current_budget):
            raise ValueError('extension requires reconciled attempts and an exhausted budget unless explicitly overridden')
        if any(_accepted(c, r) for r in rows):
            raise ValueError('extension refused: an accepted candidate already exists')
        total = current_budget + additional_attempts
        if total > _configured_budget_limit(c):
            raise ValueError('extended attempt budget exceeds configured campaign capacity')
        _append(journal, dict(event='budget_extended', previous_budget=current_budget,
                              additional_attempts=additional_attempts, budget=total, authorized_at=time.time(), allow_unspent=allow_unspent))
        return total


def run_adaptive_refinement(controller, *, workers=1, budget=48, proposer=None, exact_12_new=False, doe_handoff=None):
    """Optionally nominate 12 new attempts, ignoring acceptance but not errors or limits."""
    c, root = controller, Path(controller.run_root)
    if doe_handoff is not None and (c.config.get('fresh_finite_independent') is not True
            or c.config.get('doe_attempt_multiplier') != 5 or exact_12_new
            or workers != int(c.config.get('parallel_trials', 1))):
        raise ValueError('live DOE handoff requires the finite campaign and its full shared worker pool')
    doe_records = None if doe_handoff is None else doe_handoff['records']
    doe_pending = {} if doe_handoff is None else doe_handoff['pending']
    if (type(exact_12_new) is not bool or type(budget) is not int or budget < 0
            or type(workers) is not int or not 1 <= workers <= min(20 if c.config.get('fresh_finite_independent') is True else 12, int(c.config.get('parallel_trials', 1)))
            or int(c.config['campaign'].get('validation_trials', -1)) != 0
            or int(c.config.get('validation_trials', 0)) != 0):
        raise ValueError('refinement-only requires validation=0, bounded attempt budget and explicit allowed workers')
    initial_inflight = c.config.get('adaptive_initial_inflight', workers)
    if type(initial_inflight) is not int or not 1 <= initial_inflight <= workers:
        raise ValueError('adaptive_initial_inflight must be an integer in [1, workers]')
    configured_limit = _configured_budget_limit(c, doe_records=doe_records)
    if budget > configured_limit:
        raise ValueError('attempt budget exceeds configured campaign capacity')
    if 'sequential_experiment' in c.config and workers > c.config['sequential_experiment'].get('workers', 12):
        raise ValueError('workers exceed explicit sequential experiment limit')
    if proposer is None:
        if 'sequential_experiment' in c.config:
            from optimizer.sequential_surrogate import propose_next
        else:
            from optimizer.adaptive_surrogate import propose_next
        proposer = propose_next
    with _lock(root):
        c._validate_run_compatibility()
        initial_limit = configured_limit if 'sequential_experiment' in c.config else min(48, configured_limit)
        if not (root/'adaptive_proposals.jsonl').exists() and budget > initial_limit:
            raise ValueError('initial attempt budget exceeds configured limit')
        batch_path = root/'exact12_batch.json'
        from optimizer.repose_continuation import authorized_continuation, reconcile_completed_batch
        continuation=authorized_continuation(c)
        if batch_path.exists() and not continuation:
            raise RuntimeError('exact12 batch already persisted; explicit reconciliation required')
        with c._csv_lock:
            rows, reserved, terminal, attempted, next_id = _restore(c, budget, doe_records=doe_records)
        if doe_handoff is not None:
            # Include failed/rejected and live DOE proposals in distance checks,
            # not just already measured points. IDs cover every durable reservation.
            attempted.extend(doe_handoff['attempted'])
            next_id = max(next_id, doe_handoff['next_id'])
        if batch_path.exists() and continuation:
            proof=reconcile_completed_batch(root,reserved,terminal)
            (root/'repose_continuation.json').write_text(json.dumps(proof,indent=2))
        review_limit=budget if continuation or c.config.get('doe_attempt_multiplier') == 5 else min(budget,303)
        invocation_target = len(reserved) + 12 if exact_12_new else None
        if exact_12_new and invocation_target > review_limit:
            raise ValueError('exact 12 new attempts exceed remaining budget or review cap')
        journal = root/'adaptive_proposals.jsonl'
        if not journal.exists():
            _append(journal, dict(event='initialized', version=1, budget=budget, configuration=_settings(c)))
        if exact_12_new:
            # Write-ahead guard under the same controller lock. A crash at ANY later
            # point requires explicit reconciliation; retries never derive +12 again.
            with batch_path.open('x', encoding='utf8') as handle:
                json.dump(dict(version=1, start_reservations=len(reserved), target_reservations=invocation_target,
                               first_trial_id=next_id, requested_attempts=12, created_at=time.time(),
                               configuration=_settings(c), status='claimed_requires_reconciliation_on_retry'), handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            _sync_directory(root)
        active = {} if doe_handoff is None else doe_handoff['active']
        # Historical terminal rows never release this invocation's startup gate.
        startup_feedback_received = False
        max_inflight_observed = 0
        def summary(status, error=None):
            observed = {int(r['trial_id']) for r in rows}
            reporting_rows = rows + [dict(e['parameters'], trial_id=tid, phase='refinement', status='reserved')
                                     for tid, e in reserved.items() if tid not in observed]
            write_outcomes(c, reporting_rows, root/'adaptive_outcomes.csv')
            candidates = [int(r['trial_id']) for r in rows if _accepted(c, r)]
            report = dict(status=status, attempts=len(reserved), budget=budget,
                          completed_attempts=sum(e['event'] == 'completed' for e in terminal.values()),
                          failed_attempts=sum(e['event'] == 'failed' for e in terminal.values()),
                          candidate_trial_ids=candidates, candidate_found=bool(candidates),
                          repeatability_validation_performed=False, physical_validation_performed=False,
                          primary_outcomes_csv='adaptive_outcomes.csv', legacy_results_csv='results.csv',
                          outcome_interpretation_version=OUTCOME_VERSION,
                          configuration=_settings(c),
                          convergence_guaranteed=False, workers=workers,
                          requested_inflight_limit=workers, initial_inflight_limit=initial_inflight,
                          exact_12_new=exact_12_new, invocation_target=invocation_target,
                          startup_feedback_received=startup_feedback_received,
                          max_inflight_observed=max_inflight_observed)
            if error is not None:
                report['error'] = repr(error)
            _save_summary(root, report)
            return report
        try:
            summary('running')
            pool = ThreadPoolExecutor(max_workers=workers) if doe_handoff is None else nullcontext(doe_handoff['executor'])
            with pool as executor:
                pending = {}

                def ingest_doe(ready):
                    nonlocal rows
                    if not ready:
                        return
                    doe_handoff['ingest'](ready)
                    # Workers only append. Reload under the writer lock; preserve
                    # derived failed-refinement rows that have no measured CSV row.
                    with c._csv_lock:
                        measured = score_rows(c, [r for r in c._load_results_csv() if r.get('phase') == 'doe'])
                    # Refinement futures are ingested only by their own journal
                    # path. Reloading them here could duplicate a concurrent row.
                    rows = measured + [r for r in rows if r.get('phase') != 'doe']
                    summary('running')
                    if doe_handoff['failed']():
                        raise RuntimeError('technical error in in-flight DOE; draining campaign')

                def all_pending():
                    return set(pending) | set(doe_pending)

                def ingest(ready):
                    nonlocal startup_feedback_received
                    errors = []
                    for future in sorted(ready, key=lambda f: pending[f]):
                        tid = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = dict(reserved[tid]['parameters'], trial_id=tid, phase='refinement', status='failed')
                            outcome = dict(event='failed', trial_id=tid, error=repr(exc))
                            if exact_12_new or c.config.get('fresh_finite_independent') is True:
                                from .calibration_controller import MeasurementRejected
                                if not isinstance(exc, MeasurementRejected):
                                    errors.append(exc)
                        else:
                            try:
                                try:
                                    if (int(result['trial_id']) != tid or result.get('phase') != 'refinement'
                                            or _point(c, result) != reserved[tid]['parameters']):
                                        raise ValueError('mismatch')
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError('returned result identity/parameters mismatch') from exc
                                try:
                                    json.dumps(result, allow_nan=False)
                                    if result.get('status') not in ('complete', 'failed'):
                                        raise ValueError('invalid status')
                                    for k in c.config['tests']:
                                        value = result.get(k+'_deg')
                                        if value not in (None, '') and not math.isfinite(float(value)):
                                            raise ValueError('nonfinite response')
                                        if (result.get(k+'_response_status') == 'measured'
                                                and str(result.get(k+'_measurement_valid')).lower() == 'true'
                                                and value in (None, '')):
                                            raise ValueError('missing measured response')
                                except (TypeError, ValueError) as exc:
                                    raise ValueError('returned result payload invalid') from exc
                            except ValueError as exc:
                                errors.append(exc)
                                result = dict(reserved[tid]['parameters'], trial_id=tid, phase='refinement', status='failed')
                                outcome = dict(event='failed', trial_id=tid, error=repr(exc), error_kind='invalid_result')
                            else:
                                outcome = dict(event='completed' if result['status'] == 'complete' else 'failed',
                                               trial_id=tid, result=result)
                                if (exact_12_new or c.config.get('fresh_finite_independent') is True) and result['status'] == 'failed':
                                    errors.append(RuntimeError(f'trial {tid} returned failed status'))
                        _append(journal, dict(outcome, finished_at=time.time()))
                        terminal[tid] = outcome
                        rows.extend(score_rows(c, [result]))
                        startup_feedback_received = True  # Durable and ingested, including technical failure.
                        c._record_status(trial_id=tid, phase='refinement', status=outcome['event'], stage='adaptive',
                                         message=outcome.get('error', 'result ingested'))
                        active.pop(tid)
                    summary('running')
                    if errors:
                        raise errors[0]

                try:
                    while True:
                        from .transition_guards import drain_requested
                        ready_doe = {f for f in doe_pending if f.done()}
                        if ready_doe:
                            ingest_doe(ready_doe)
                            continue
                        ready = {f for f in pending if f.done()}
                        if ready:
                            ingest(ready)
                            continue  # Re-scan completions before ANY nomination or refill.
                        stop = (drain_requested(root) or len(reserved) >= (invocation_target if exact_12_new else review_limit)
                                or (not exact_12_new and any(_accepted(c, r) for r in rows)))
                        if not all_pending() and stop:
                            break
                        effective_inflight_limit = workers if startup_feedback_received else initial_inflight
                        if not stop and len(all_pending()) < effective_inflight_limit:
                            point, diagnostics = proposer(c, rows, attempted, proposal_index=len(reserved))
                            if drain_requested(root):
                                continue  # Discard nominations prepared before the drain request.
                            if any(f.done() for f in all_pending()):
                                continue  # A result arrived during fitting: discard and refit.
                            point = _point(c, point, exact=True)
                            minimum = float(c.config.get('adaptive_min_distance', .01))
                            distance = _distance(c, point, attempted)
                            if not math.isfinite(minimum) or minimum < .01 or distance < minimum:
                                raise ValueError('proposal violates minimum normalized distance')
                            tid, next_id = next_id, next_id+1
                            event = dict(event='reserved', trial_id=tid, parameters=point, model_revision=len(rows),
                                         observed_result_ids=[int(r['trial_id']) for r in rows], diagnostics=diagnostics,
                                         pending_trial_ids=sorted(list(pending.values()) + [v[0] for v in doe_pending.values()]),
                                         effective_inflight_limit=effective_inflight_limit,
                                         startup_feedback_received=startup_feedback_received,
                                         min_distance=distance, seeds=c._trial_seeds(tid), reserved_at=time.time())
                            _append(journal, event)  # Full reservation is fsynced before submit.
                            reserved[tid] = event
                            attempted.append(point)
                            c._record_status(trial_id=tid, phase='refinement', status='reserved', stage='adaptive',
                                             message='durable adaptive reservation')
                            active[tid] = dict(phase='refinement', status='preparing', started_at=time.time())
                            pending[executor.submit(c.execute_trial, tid, point, 'refinement', event['seeds'])] = tid
                            max_inflight_observed = max(max_inflight_observed, len(all_pending()))
                            write_live_status_csv(root, active)
                            summary('running')
                            continue
                        refresh_live_state(root, active)
                        write_live_status_csv(root, active)
                        render_dashboard(active, len(terminal), budget)
                        wait(all_pending(), timeout=10, return_when=FIRST_COMPLETED)
                        ingest_doe({f for f in doe_pending if f.done()})
                        ingest({f for f in pending if f.done()})
                except BaseException:
                    # Drain and journal workers even when proposing/status updates fail.
                    # No cancellation and no results.csv rewrite while a worker exists.
                    while all_pending():
                        wait(all_pending(), timeout=10, return_when=FIRST_COMPLETED)
                        try:
                            ingest_doe({f for f in doe_pending if f.done()})
                        except Exception:
                            pass  # Outcomes are durable; preserve the original error.
                        try:
                            ingest({f for f in pending if f.done()})
                        except Exception:
                            # Each invalid result is already journaled; preserve the
                            # exception that originally triggered draining.
                            continue
                    raise
            if not exact_12_new and c.config.get('doe_attempt_multiplier') != 5:
                _finish_scores(c)  # Exact restart preserves all pre-existing rows verbatim.
            if drain_requested(root):
                return summary('drained_for_transition')
            return summary('exact_12_completed' if exact_12_new else ('candidate_found' if any(_accepted(c, r) for r in rows) else ('review_cap_reached' if len(reserved) >= review_limit and budget > review_limit else 'budget_exhausted')))
        except BaseException as exc:
            summary('interrupted' if not isinstance(exc, Exception) else 'error', error=exc)
            raise
        finally:
            write_live_status_csv(root, {})
