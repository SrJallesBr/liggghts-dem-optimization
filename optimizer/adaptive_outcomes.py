"""Interpret legacy evidence without changing it or assigning measured angles."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from optimizer.article_optimizer import weighted_normalized_score

VERSION = 'adaptive-outcomes-v3'
OUTCOME_CODES = ('measured', 'retained_column', 'too_cohesive',
                 'measurement_unresolved', 'technical_failure', 'pending')


def _true(value):
    return strict_bool(value) is True


def strict_bool(value):
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.lower() in ('true','false'):
        return value.lower() == 'true'
    return None


def structured_evidence(value):
    """Only JSON objects/arrays containing real evidence are ingestible."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, (dict,list)) or not value:
        return None
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return None
    def substantive(item):
        if isinstance(item, dict): return any(substantive(v) for v in item.values())
        if isinstance(item, list): return any(substantive(v) for v in item)
        return isinstance(item, str) and bool(item.strip()) or type(item) in (int,float)
    return value if substantive(value) else None


def response_row(row, kind):
    """Read both canonical flat rows and the controller's lossless trial.json."""
    if not isinstance(row.get('responses'), dict) or kind not in row['responses']:
        return row
    raw = row['responses'][kind]
    if not isinstance(raw, dict) or not isinstance(raw.get('optimization_response', {}), dict):
        return dict(row, **{kind+'_raw_status':'invalid_data'})
    response = raw.get('optimization_response', {})
    result = dict(row)
    for key, value in dict(raw_deg=raw.get('response_angle_deg'), raw_status=raw.get('status'),
            raw_reason=raw.get('reason'), response_status=response.get('adopted_status'),
            measurement_valid=response.get('measurement_valid'), assignment_reason=response.get('assignment_reason'),
            deg=response.get('adopted_angle_deg')).items():
        field = kind+'_'+key
        if field in result and result[field] != value:
            result[kind+'_raw_status'] = 'invalid_data'
            result[kind+'_raw_reason'] = 'Contradictory flat/nested response metadata'
            return result
        result.setdefault(field, value)
    for suffix in ('physical_outcome','physical_outcome_verified','physical_outcome_evidence'):
        if suffix in response or suffix in raw:
            field, value = kind+'_'+suffix, response.get(suffix, raw.get(suffix))
            if field in result and result[field] != value:
                result[kind+'_raw_status'] = 'invalid_data'
                result[kind+'_raw_reason'] = 'Contradictory flat/nested physical metadata'
                return result
            result.setdefault(field, value)
    return result


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def interpret_response(row, kind, target=None):
    """A physical label is known only for measured or explicit physical evidence.

    Retained-column geometry is a local snapshot heuristic, never a verified
    causal diagnosis. Missing/failed fits provide no physical classifier label.
    """
    row = response_row(row, kind)
    def get(suffix):
        return row.get(kind+'_'+suffix)
    angle = _finite(get('deg'))
    raw_reason, assignment = get('raw_reason'), get('assignment_reason')
    canonical = kind+'_physical_outcome_verified' in row or kind+'_physical_outcome_evidence' in row
    evidence = structured_evidence(get('physical_outcome_evidence') if canonical else get('physical_evidence'))
    verified_flag = strict_bool(get('physical_outcome_verified') if canonical else get('physical_diagnosis_verified'))
    raw_status = get('raw_status')
    raw_ok = kind+'_raw_status' not in row or raw_status in ('', 'measured', 'ok')
    # Absent historical raw metadata is compatible; explicit bad metadata vetoes.
    raw_angle_ok = True
    if kind+'_raw_deg' in row and (raw_status not in (None,'') or get('raw_deg') not in (None,'')):
        raw_angle = _finite(get('raw_deg'))
        raw_angle_ok = (raw_angle is not None and angle is not None
                        and math.isclose(raw_angle, angle, rel_tol=0., abs_tol=1e-9))
    # Measurement/adoption flags are mandatory; only raw metadata may be absent
    # in legacy rows. CSV blanks are unknown, never affirmative validity.
    response_ok = get('response_status') == 'measured'
    valid_ok = _true(get('measurement_valid'))
    code, label, basis, physical, verified = (
        'measurement_unresolved', 'Mesuração não resolvida', 'no_valid_angle', None, False)
    if (row.get('exception') or row.get('error')
            or get('raw_status') in ('error', 'failed', 'exception')):
        code, label, basis = 'technical_failure', 'Falha técnica', 'execution_status_or_exception'
    elif row.get('status') in ('reserved', 'running', 'preparing', 'queued'):
        code, label, basis = 'pending', 'Aguardando medição', 'in_flight_execution_status'
    elif row.get('status') != 'complete':
        code, label, basis = 'technical_failure', 'Falha técnica', 'execution_status_or_exception'
    elif (response_ok and valid_ok and angle is not None and raw_ok and raw_angle_ok):
        code, label, basis, physical = 'measured', 'Ângulo medido', 'measurement_flags_and_finite_angle', 1
    elif (get('physical_outcome') == 'too_cohesive' and verified_flag is True and evidence is not None
          and raw_status in (None,'','unmeasurable','measured','ok')):
        code, label, basis, physical, verified = (
            'too_cohesive', 'Coesivo demais — diagnóstico confirmado', 'explicit_verified_evidence', 0, True)
    elif get('raw_status') == 'unmeasurable' and (
            str(raw_reason or '').startswith('Retained column:')
            or str(assignment or '').startswith('Retained column geometry')):
        code, label, basis, physical = (
            'retained_column', 'Coluna retida — comportamento coesivo aparente', 'local_snapshot_heuristic', 0)
    measured = code == 'measured'
    return dict(outcome_interpretation_version=VERSION, outcome_code=code, label_pt=label, basis=basis,
                angle_deg=angle if measured else None,
                error_deg=abs(angle-target) if measured and target is not None else None,
                regression_included=measured, physical_classifier_label=physical,
                causal_cohesion_verified=verified, legacy_adopted_deg=angle,
                legacy_response_status=get('response_status'), measurement_valid=get('measurement_valid'),
                raw_deg=get('raw_deg'), raw_status=get('raw_status'), raw_reason=raw_reason,
                assignment_reason=assignment, physical_outcome=get('physical_outcome'),
                physical_diagnosis_verified=verified_flag, physical_evidence=evidence,
                physical_outcome_verified=verified_flag, physical_outcome_evidence=evidence)


def write_outcomes(controller, rows, path):
    """Atomically publish one primary reporting row per trial/test from memory.

    No legacy result file is read or modified. Call under the scheduler lock.
    An all-measured trial objective is the only score exposed in this CSV.
    """
    from optimizer.protocol_eligibility import interpret_for_controller
    c, path = controller, Path(path)
    records = []
    for row in rows:
        outcomes = {k: interpret_for_controller(c, row, k, c.config['target_'+k+'_deg']) for k in c.config['tests']}
        score = None
        if all(o['regression_included'] for o in outcomes.values()):
            score = weighted_normalized_score({k: o['error_deg'] for k, o in outcomes.items()},
                                              c.response_scales, c.config['weights'])
        provenance = {key: row.get(key) for key in (
            'trial_id', 'phase', 'status', 'workspace', 'measurement_method', 'response_policy',
            'distribution_seed', 'insertion_seed', 'exception', 'error')}
        provenance['parameters_json'] = json.dumps({n: row.get(n) for n in c.parameter_names}, sort_keys=True)
        for kind, outcome in outcomes.items():
            record = dict(provenance, test=kind, **outcome, all_measured_score=score)
            for field in ('physical_evidence','physical_outcome_evidence'):
                record[field] = json.dumps(record[field], ensure_ascii=False, allow_nan=False)
            records.append(record)
    fields = list(dict.fromkeys(key for record in records for key in record)) if records else [
        'trial_id', 'test', 'outcome_code', 'angle_deg', 'error_deg', 'legacy_adopted_deg']
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
