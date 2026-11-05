"""Isolated feedback-guided DOE selection (stdlib + NumPy, no I/O).

Only joint measurability labels are modelled, never response angles or targets.
Caller must supply observations from the SAME geometry/protocol. Include every
attempt/reservation in observed; eligible=None excludes it from model fitting
but not distance checks. Optional state='pending' or 'history' distinguishes
unlabelled reservations from technical failures in the audit counts.

Exploration is label-independent maxmin over the original full-domain pool.
CED11 and CED12 upper-quartile quotas are independent and may overlap. For
one-point batches the two quotas necessarily constrain that exploration point
to their intersection; quota feasibility similarly applies near batch end.
Finite candidate exhaustion raises, never relaxes separation or returns a
partial batch. An empirical IDW score is not a calibrated probability.
"""
from collections.abc import Mapping
from itertools import product
from math import ceil
from numbers import Integral, Real

import numpy as np

PARAMETERS = ('sf11', 'rf11', 'CED11', 'sf12', 'rf12', 'CED12')
_POOL_SIZE = 8192
_NEIGHBORS = 8
_DIVERSITY_WEIGHT = .25


def _finite_number(value, description):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(description + ' must be a finite number')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(description + ' must be a finite number')
    return value


def _inputs(bounds, observed, count, seed, minimum_separation):
    if not isinstance(bounds, Mapping) or set(bounds) != set(PARAMETERS):
        raise ValueError('bounds must contain exactly the six parameter names')
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or not 0 <= count <= 12:
        raise ValueError('count must be an integer from 0 through 12')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    separation = _finite_number(minimum_separation, 'minimum_separation')
    if separation <= 0:
        raise ValueError('minimum_separation must be positive')
    try:
        limits = []
        for k in PARAMETERS:
            if len(bounds[k]) != 2:
                raise ValueError('each parameter needs exactly two bounds')
            limits.append([_finite_number(v, 'bound ' + k) for v in bounds[k]])
        limits = np.array(limits)
    except TypeError as exc:
        raise ValueError('each parameter needs exactly two numeric bounds') from exc
    lo, hi = limits[:, 0], limits[:, 1]
    if np.any(hi <= lo) or not np.all(np.isfinite(hi - lo)):
        raise ValueError('each upper bound must exceed its lower bound with finite span')
    positions, labelled_positions, labels = [], [], []
    counts = dict(total=0, eligible=0, ineligible=0, technical=0, pending=0, history=0)
    for r in observed:
        if not isinstance(r, Mapping) or 'eligible' not in r or any(k not in r for k in PARAMETERS):
            raise ValueError('each observation requires six parameters and eligible')
        label = r['eligible']
        if label is not None and not isinstance(label, (bool, np.bool_)):
            raise ValueError('eligible must be bool or None, not a truthy string/number')
        raw = np.array([_finite_number(r[k], 'observation ' + k) for k in PARAMETERS])
        if np.any(raw < lo) or np.any(raw > hi):
            raise ValueError('observation outside supplied bounds; do not mix domains')
        pos = (raw - lo) / (hi - lo)
        positions.append(pos)
        counts['total'] += 1
        if label is not None:
            counts['eligible' if bool(label) else 'ineligible'] += 1
            labelled_positions.append(pos)
            labels.append(float(label))
        else:
            state = r.get('state')
            counts[state if state in ('pending', 'history') else 'technical'] += 1
    return (lo, hi, np.array(positions).reshape(-1, 6),
            np.array(labelled_positions).reshape(-1, 6), np.array(labels),
            counts, int(count), int(seed), separation)


def _predict_measurability(candidates, labelled_positions, labels):
    """Eight-nearest IDW using Euclidean distance in ALL six normalized axes.

    Technical failures/reservations must already have been removed from labels.
    Exact repeated labelled positions use the mean of all coincident labels.
    Chunking bounds temporary memory; stable neighbor sorting resolves ties.
    """
    if not len(labels):
        return np.full(len(candidates), np.nan)
    result = np.empty(len(candidates))
    for start in range(0, len(candidates), 256):
        c = candidates[start:start + 256]
        distances = np.sqrt(np.sum((c[:, None, :] - labelled_positions[None, :, :]) ** 2, axis=2))
        idx = np.argsort(distances, axis=1, kind='stable')[:, :min(_NEIGHBORS, len(labels))]
        near = np.take_along_axis(distances, idx, axis=1)
        weights = 1. / np.maximum(near, 1e-12)
        prediction = (weights * labels[idx]).sum(axis=1) / weights.sum(axis=1)
        for j in np.flatnonzero(near[:, 0] <= 1e-12):
            prediction[j] = labels[distances[j] <= 1e-12].mean()
        result[start:start + len(c)] = prediction
    return result


def _minimum_distances(candidates, history):
    result = np.full(len(candidates), np.inf)
    for start in range(0, len(history), 128):
        refs = history[start:start + 128]
        d = np.abs(candidates[:, None, :] - refs[None, :, :]).max(axis=2).min(axis=1)
        result = np.minimum(result, d)
    return result


def select_points(bounds, observed, count, seed, minimum_separation=0.05):
    """Return (list of six-key parameter dicts, JSON-serializable audit).

    Bounds are {name: (low, high)}. No mutation, persistence, campaign loading,
    solver execution, angle fabrication, scalar cohesion cutoff or bound change.
    Linf exclusion is checked against ALL rows, including None/pending/history.
    >=ceil(count/3) choices use label-independent full-pool maxmin; >=ceil(count/4)
    choices lie in EACH upper CED quartile. Remaining choices maximize empirical
    measurability + 0.25 * normalized Linf novelty. With no eligible=True data,
    every choice uses maxmin instead. High-quartile requirements remain active.
    """
    lo, hi, history, training, labels, counts, count, seed, separation = _inputs(
        bounds, observed, count, seed, minimum_separation)
    n_explore, quota = ceil(count / 3), ceil(count / 4)
    guided = counts['eligible'] > 0
    audit = dict(
        policy=dict(name='six_dimensional_idw_diverse_backfill', version=1,
                    normalization='original_bound_spans', neighbors=_NEIGHBORS,
                    prediction_metric='normalized_6d_euclidean', inverse_distance_power=1,
                    separation_metric='normalized_6d_Linf', diversity_weight=_DIVERSITY_WEIGHT,
                    exploration_required=n_explore, high_CED11_required=quota,
                    high_CED12_required=quota,
                    small_batch_note='quotas may constrain full-pool maxmin; count=1 requires both high quartiles',
                    prediction_note='empirical measurability score, not a calibrated probability'),
        seed=seed, parameter_bounds={k: [float(lo[i]), float(hi[i])] for i, k in enumerate(PARAMETERS)},
        observation_counts=counts, minimum_separation=separation,
        fallback=None if guided else 'maxmin_full_bounds', proposals=[],
        coverage=dict(exploration=0, high_CED11=0, high_CED12=0))
    if count == 0:
        return [], audit
    rng = np.random.default_rng(seed)
    # Corners make boundary/maxmin exploration possible, not only interior LHS.
    candidates = np.vstack([rng.random((_POOL_SIZE, 6)), np.array(list(product((0., 1.), repeat=6)))])
    candidates = candidates[rng.permutation(len(candidates))]
    # Round-trip to output units before checking distances (no boundary tolerance
    # can silently accept a normalized separation smaller than the requested one).
    physical = lo + candidates * (hi - lo)
    candidates = (physical - lo) / (hi - lo)
    predictions = _predict_measurability(candidates, training, labels)
    nearest = _minimum_distances(candidates, history)
    high11, high12 = candidates[:, 2] >= .75, candidates[:, 5] >= .75
    used11 = used12 = 0
    points = []
    for slot in range(count):
        remaining = count - slot - 1
        feasible = nearest >= separation
        # Leave enough slots to satisfy BOTH independent quotas; the same future
        # point may satisfy both. If finite pool cannot do this, fail explicitly.
        feasible &= np.maximum(0, quota - used11 - high11.astype(int)) <= remaining
        feasible &= np.maximum(0, quota - used12 - high12.astype(int)) <= remaining
        if slot < n_explore:
            mode, reason = 'exploration', 'full_domain_maxmin'
        else:
            mode = 'feedback' if guided else 'maxmin'
            reason = 'feedback_idw_diversity' if guided else 'full_domain_maxmin'
            if used11 < quota:
                feasible &= high11
                reason = 'high_CED11_' + ('idw_diversity' if guided else 'maxmin')
            elif used12 < quota:
                feasible &= high12
                reason = 'high_CED12_' + ('idw_diversity' if guided else 'maxmin')
        indices = np.flatnonzero(feasible)
        if not len(indices):
            raise ValueError('insufficient candidate diversity/separation for complete batch and high-quartile quotas; no partial batch returned')
        novelty = nearest.copy()
        if not np.isfinite(novelty).any():
            # Initial, unobserved design: maxmin against a virtual centre used
            # for scoring only, never misreported as an attempted observation.
            novelty = np.abs(candidates - .5).max(axis=1)
        score = novelty
        if mode == 'feedback':
            score = predictions + _DIVERSITY_WEIGHT * novelty
        index = int(indices[np.argmax(score[indices])])
        point = {k: float(physical[index, i]) for i, k in enumerate(PARAMETERS)}
        points.append(point)
        prediction = float(predictions[index]) if len(labels) else None
        audit['proposals'].append(dict(
            point=dict(point), mode=mode, reason=reason, prediction=prediction,
            minimum_separation=float(nearest[index]) if np.isfinite(nearest[index]) else None,
            high_CED11=bool(high11[index]), high_CED12=bool(high12[index])))
        used11 += int(high11[index])
        used12 += int(high12[index])
        if mode == 'exploration':
            audit['coverage']['exploration'] += 1
        nearest = np.minimum(nearest, np.abs(candidates - candidates[index]).max(axis=1))
    audit['coverage'].update(high_CED11=used11, high_CED12=used12)
    audit['candidate_pool_size'] = len(candidates)
    return points, audit
