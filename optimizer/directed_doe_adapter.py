"""Translate real DOE measurement/QC records into audited six-dimensional proposals."""
from pathlib import Path
import json
import math
from optimizer.adaptive_outcomes import interpret_response


class ProposalPoints(list):
    """List-compatible proposals with a publication-snapshot revision."""
    def __init__(self, points, revision):
        super().__init__(points)
        self.observation_revision = revision


def build_observations(controller, points):
    names = controller.parameter_names
    def params(row):
        try:
            p = {n:float(row[n]) for n in names}
        except (ValueError, TypeError, KeyError):
            return None
        return p if all(math.isfinite(v) for v in p.values()) else None
    def key(p):
        return tuple(round(p[n], 12) for n in names)
    history = {}
    def reserve(p, reason):
        p = params(p)
        if p is not None:
            history.setdefault(key(p), {**p, 'eligible':None, 'observation_reason':reason,
                'state':'pending' if reason == 'planned_initial_point' else 'history'})
    for p in points:
        reserve(p, 'planned_initial_point')
    # All reservations prevent repeated/near repeated points even when no results
    # were committed. Unrun or technically failed points supply NO physics label.
    paths = list(controller.run_root.glob('trial_*/parameters.json')) + list(controller.run_root.glob('trial_*/trial_parameters.json'))
    for path in sorted(paths):
        meta = json.loads(path.read_text())
        if meta.get('phase', 'doe') == 'doe':
            reserve(meta.get('parameters', meta), 'reserved_or_failed_attempt')
    diagnostics = controller.run_root/'diagnostics.csv'
    if diagnostics.exists():
        import csv
        with diagnostics.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('phase') == 'doe' and row.get('parameters_json'):
                    point = params(json.loads(row['parameters_json']))
                    reserve(point, 'diagnostic_attempt')
                    if point is not None and row.get('status') == 'geometry_unmeasurable':
                        history[key(point)] = dict(point, eligible=False, observation_reason='geometry_diagnostic')
    for row in controller._load_results_csv():
        if row.get('phase') != 'doe':
            continue
        p = params(row)
        if p is None:
            continue
        interpreted = {kind:interpret_response(row, kind) for kind, *_ in controller.tests}
        valid = controller._valid_responses(row)
        classes = [v['outcome_code'] for v in interpreted.values()]
        eligible = True if valid else (None if 'technical_failure' in classes else False)
        history[key(p)] = dict(**p, eligible=eligible, observation_reason='measured_qc',
            response_classifications={k:v['outcome_code'] for k,v in interpreted.items()})
    return list(history.values())


def select_for_controller(controller, points, count, seed, pending_points=()):
    """Nominate information-seeking physical points; pending points carry no label.

    Interface shared with optimizer.informative_doe: physical parameter dictionaries,
    raw result CSV rows, unchanged physical bounds, explicit count/pending/seed.
    """
    import csv
    import os
    from contextlib import nullcontext
    from optimizer.informative_doe import select_informative_points

    names = controller.parameter_names
    def physical(p):
        return {n: float(p[n]) for n in names}
    def key(p):
        return tuple(round(float(p[n]), 12) for n in names)
    with getattr(controller, '_csv_lock', nullcontext()):
        revision = getattr(controller, '_results_revision', 0)
        history = build_observations(controller, points)
        path = controller.run_root/'results.csv'
        with path.open(newline='', encoding='utf8') if path.exists() else nullcontext(None) as handle:
            outcomes = [dict(r) for r in csv.DictReader(handle) if r.get('phase') == 'doe'] if handle else []
        journal = controller.run_root/'doe_schedule.jsonl'
        reservations, terminal = {}, set()
        if journal.exists():
            for line in journal.read_text(encoding='utf8').splitlines():
                event = json.loads(line)
                tid = int(event['trial_id'])
                if event['event'] == 'reserved':
                    reservations[tid] = physical(event['parameters'])
                else:
                    terminal.add(tid)
        # CSV-complete outcomes can precede scheduler ingestion by a few ms.
        terminal.update(int(r['trial_id']) for r in outcomes if r.get('status') == 'complete')
        existing = {key(p): physical(p) for p in history}
        existing.update({key(p): p for p in reservations.values()})
        pending = {key(p): physical(p) for p in pending_points}
        pending.update({key(p): p for tid, p in reservations.items() if tid not in terminal})
    bounds = {n: (controller.config['parameters'][n]['min'], controller.config['parameters'][n]['max']) for n in names}
    selected = select_informative_points(existing_points=list(existing.values()), outcomes=outcomes,
        bounds=bounds, count=count, pending_points=list(pending.values()), seed=seed,
        responses=controller.config.get('tests',['discharge','repose']))
    proposals, audit = selected['points'], dict(selected['diagnostics'])
    if len(proposals) != count:
        raise RuntimeError('informative picker did not return requested number of points')
    used = set(existing) | set(pending)
    for p in proposals:
        if (set(p) != set(names) or any(not math.isfinite(float(p[n])) or not bounds[n][0] <= float(p[n]) <= bounds[n][1] for n in names)):
            raise ValueError('informative DOE proposal outside physical bounds')
        if key(p) in used:
            raise ValueError('duplicate informative DOE proposal (history or pending)')
        used.add(key(p))
    audit.update(source='raw completed trial CSV responses/QC and physical reservations; no inferred angles',
        source_planned_points=len(points), history_scored_results=sum(o['observation_reason']=='measured_qc' for o in history),
        configured_bounds=bounds, original_boundaries_unchanged=True,
        feedback_observations=history, pending_points=list(pending.values()),
        observation_revision=revision, supported_responses=controller.config.get('tests',['discharge','repose']),
        proposals=proposals, actual_doe_attempts=controller._doe_attempt_count())
    dest = controller.run_root/('doe_selection_audit_'+str(seed)+'.json')
    if dest.exists():
        raise RuntimeError('refusing to overwrite selection audit '+str(dest))
    temp = dest.with_suffix('.tmp')
    with temp.open('w', encoding='utf8') as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(dest)
    return ProposalPoints(proposals, revision)
