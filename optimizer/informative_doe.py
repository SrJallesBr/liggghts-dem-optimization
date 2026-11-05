"""Pure six-factor informative DOE selection; no solver or persistence.

All normalization uses the caller's unchanged physical bounds. Design information
is prospective (attempts/reservations are NOT measured response observations).
"""
from collections.abc import Mapping
from itertools import combinations, product
from numbers import Integral

import numpy as np

PARAMETERS = ('sf11', 'rf11', 'CED11', 'sf12', 'rf12', 'CED12')
_PAIRS = tuple(combinations(range(6), 2))
_RIDGE = 1e-3
_MIN_DISTANCE = 1e-6
_POOL_SIZE = 2048
_FEATURE_COUNT = 28


def _quadratic_features(x):
    """28 columns: intercept, six centered mains, six squares, 15 interactions."""
    z = 2*np.asarray(x, dtype=float).reshape(-1, 6) - 1
    return np.column_stack([np.ones(len(z)), z, z*z,
                            *[z[:, i]*z[:, j] for i, j in _PAIRS]])


def _limits(bounds):
    if not isinstance(bounds, Mapping) or set(bounds) != set(PARAMETERS):
        raise ValueError('bounds require exactly '+', '.join(PARAMETERS))
    try:
        if any(len(bounds[k]) != 2 or any(isinstance(v, (bool, np.bool_)) for v in bounds[k])
               for k in PARAMETERS):
            raise ValueError('bounds require two finite numbers per parameter')
        a = np.array([bounds[k] for k in PARAMETERS], float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('bounds require two finite numbers per parameter') from exc
    low, high = a[:, 0], a[:, 1]
    if not np.isfinite(a).all() or np.any(high <= low) or not np.isfinite(high-low).all():
        raise ValueError('bounds require finite positive spans')
    return low, high


def _position(row, low, high):
    try:
        if any(isinstance(row[k], (bool, np.bool_)) for k in PARAMETERS):
            raise ValueError('boolean parameter')
        p = np.array([row[k] for k in PARAMETERS], float)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('parameters require all six finite physical values') from exc
    if not np.isfinite(p).all() or np.any(p < low) or np.any(p > high):
        raise ValueError('parameters outside bounds or nonfinite')
    return (p-low)/(high-low)


def _unique(x):
    a = np.asarray(x, float).reshape(-1, 6)
    return np.unique(a, axis=0)


def _minimum_distances(pool, refs):
    """Bounded temporary pool-by-64 blocks, never pool-by-pool matrices."""
    result = np.ones(len(pool))
    for start in range(0, len(refs), 64):
        diff = pool[:, None, :] - refs[None, start:start+64, :]
        result = np.minimum(result, np.sqrt(np.sum(diff*diff, axis=2)/6).min(axis=1))
    return result


def _candidate_pool(rng):
    lhs = np.column_stack([(rng.permutation(_POOL_SIZE)+rng.random(_POOL_SIZE))/_POOL_SIZE
                           for _ in PARAMETERS])
    return _unique(np.vstack([lhs, np.array(list(product((0., 1.), repeat=6))),
                              np.full((1, 6), .5)]))


def _design(f):
    # Only 28x28 linear algebra, regardless of candidate count.
    matrix = f.T@f
    eigenvalues, vectors = np.linalg.eigh(matrix)
    null = vectors[:, eigenvalues < 1e-9]
    return np.linalg.inv(matrix+_RIDGE*np.eye(28)), null, 28-null.shape[1]


def _obstruction_candidates(centers):
    """Counterfactual probes, NOT a causal/cohesion diagnosis or narrowed domain.

    Every factor moves independently and every pair jointly, including both
    cohesions and each friction/cohesion interaction. No high-cohesion quota.
    """
    result = []
    centers = np.asarray(centers, dtype=float).reshape(-1, 6)
    for center in centers:
        for i in range(6):
            for sign in (-1, 1):
                x = center.copy()
                x[i] += sign*.15
                result.append(np.clip(x, 0., 1.))
        for i in range(6):
            for j in range(i+1, 6):
                for a, b in product((-.15, .15), repeat=2):
                    x = center.copy()
                    x[i] += a
                    x[j] += b
                    result.append(np.clip(x, 0., 1.))
    return _unique(np.array(result).reshape(-1, 6))


def _feasibility_probability(pool, label_points, labels):
    """Conservative beta(2,2) local evidence estimate, never a calibrated claim.

    Absence of obstruction evidence is not failure; technical/unresolved cases
    never enter labels. Distances use all six factors without sign assumptions.
    Only pool-by-small-evidence chunks are allocated, never pool-by-pool.
    """
    prior_success = np.full(len(pool), 2.)
    prior_total = np.full(len(pool), 4.)
    for start in range(0, len(label_points), 64):
        points, y = label_points[start:start+64], labels[start:start+64]
        d2 = np.sum((pool[:, None, :]-points[None, :, :])**2, axis=2)
        weights = np.exp(-d2/(2*.25**2))
        prior_success += weights@y
        prior_total += weights.sum(axis=1)
    return np.clip(prior_success/prior_total, .15, .85)


def _evidence(outcomes, limits, kinds):
    # Delegate acceptance to the existing QC interpreter; never reclassify a
    # bad fit, invalid angle, generic failure, or assigned 90 as cohesion.
    from optimizer.adaptive_outcomes import interpret_response
    counts = dict(raw_rows=len(outcomes), valid_parameter_rows=0, invalid_parameter_rows=0,
                  technical_rows=0, pending_rows=0, physical_obstruction_rows=0,
                  measured_eligible_rows=0, unresolved_rows=0, physical_label_points=0,
                  conflicting_label_points=0)
    design, grouped = [], {}
    response_groups = {kind:{} for kind in kinds}
    for row in outcomes:
        try:
            x = _position(row, *limits)
        except (ValueError, TypeError):
            counts['invalid_parameter_rows'] += 1
            continue
        counts['valid_parameter_rows'] += 1
        design.append(x)
        responses = [interpret_response(row, kind) for kind in kinds]
        codes = {r['outcome_code'] for r in responses}
        if 'technical_failure' in codes:
            counts['technical_rows'] += 1
            continue
        if 'pending' in codes:
            counts['pending_rows'] += 1
            continue
        for kind, response in zip(kinds, responses):
            if response['regression_included']:
                key = tuple(x.round(10))
                item = response_groups[kind].setdefault(key, dict(point=x, values=[]))
                item['values'].append(response['angle_deg'])
        measured = all(r['regression_included'] for r in responses)
        obstructed = bool(codes & {'retained_column', 'too_cohesive'})
        if obstructed:
            counts['physical_obstruction_rows'] += 1
            label = 0.
        elif measured:
            counts['measured_eligible_rows'] += 1
            label = 1.
        else:
            counts['unresolved_rows'] += 1
            continue
        key = tuple(x.round(10))
        item = grouped.setdefault(key, dict(point=x, labels=set(), responses=[]))
        item['labels'].add(label)
        if measured:
            item['responses'].append([r['angle_deg'] for r in responses])
    labeled, values, measured_points, targets = [], [], [], []
    for item in grouped.values():
        if len(item['labels']) != 1:
            counts['conflicting_label_points'] += 1
            continue  # Contradictory repeated evidence is unknown, not majority-voted.
        label = next(iter(item['labels']))
        labeled.append(item['point'])
        values.append(label)
        if label == 1. and item['responses']:
            measured_points.append(item['point'])
            targets.append(np.mean(item['responses'], axis=0))
    labeled = np.array(labeled).reshape(-1, 6)
    values = np.array(values)
    counts['physical_label_points'] = len(labeled)
    measured_points = np.array(measured_points).reshape(-1, 6)
    targets = np.array(targets).reshape(-1, len(kinds))
    per_response = {}
    for kind, groups in response_groups.items():
        per_response[kind] = dict(points=np.array([g['point'] for g in groups.values()]).reshape(-1, 6),
                                  targets=np.array([np.mean(g['values']) for g in groups.values()]))
    return dict(per_response=per_response, counts=counts, design=np.array(design).reshape(-1, 6),
                labeled=labeled, labels=values, measured_points=measured_points, targets=targets)


def _response_uncertainty(pool, evidence):
    """Independent QC-eligible responses; design coverage is NOT accuracy.

    At least 32 distinct measured points and rank 28 per response are needed.
    Fixed-ridge PRESS is analytic leave-one-out prediction error in degrees,
    not training RMSE, an optimized hyperparameter, or calibrated uncertainty.
    Only 28-column matrices are used; no sample-by-sample hat matrix is built.
    """
    scores, models, quality = [], {}, {}
    candidate_features = _quadratic_features(pool)
    for kind in evidence['per_response']:
        points = evidence['per_response'][kind]['points']
        y = evidence['per_response'][kind]['targets']
        features = _quadratic_features(points)
        _, _, rank = _design(features)
        diagnostic = dict(eligible_measured_points=len(points), design_rank=int(rank),
                          method='fixed_ridge_leave_one_out_PRESS', rmse_deg=None,
                          unavailable_reason=None, calibrated_prediction_intervals=False,
                          warning='LOOCV estimates predictive error, not a guarantee of optimality or calibration.')
        quality[kind] = diagnostic
        if len(points) < 32:
            diagnostic['unavailable_reason'] = 'insufficient_eligible_measured_points'
            continue
        if rank < _FEATURE_COUNT:
            diagnostic['unavailable_reason'] = 'quadratic_response_design_rank_deficient'
            continue
        # Unpenalized intercept makes raw-output PRESS identical to a centered
        # response fit. Output scaling uses measurements only, never assigned 90.
        penalty = np.diag([0.]+[_RIDGE]*27)
        inverse = np.linalg.inv(features.T@features+penalty)
        leverage = np.maximum(np.einsum('ij,jk,ik->i', candidate_features, inverse, candidate_features), 0.)
        raw_residuals = y-features@(inverse@features.T@y)
        hat_diagonal = np.einsum('ij,jk,ik->i', features, inverse, features)
        if np.any(1.-hat_diagonal <= 1e-8):
            diagnostic['unavailable_reason'] = 'unstable_leave_one_out_leverage'
        else:
            diagnostic['rmse_deg'] = float(np.sqrt(np.mean((raw_residuals/(1.-hat_diagonal))**2)))
        standardized_residuals = raw_residuals/max(float(y.std()), 1.)
        variance = max(float(standardized_residuals@standardized_residuals)/max(len(points)-rank, 1), .01)
        se = np.sqrt(leverage*variance)
        scores.append(se/(1.+se))
        models[kind] = dict(training_count=len(points), design_rank=rank, ridge=_RIDGE,
                            standardized_residual_variance=variance,
                            observations='measured_per_response_QC_only_no_imputation')
    return (np.mean(scores, axis=0) if scores else np.zeros(len(pool))), models, quality


def _associations(evidence):
    labels = evidence['labels']
    if np.sum(labels == 0) < 2 or np.sum(labels == 1) < 2:
        return []
    # Contrasts expose all multivariate hypotheses, never choose a sole cause.
    f = _quadratic_features(evidence['labeled'])
    contrast = (f[labels == 0].mean(axis=0)-f[labels == 1].mean(axis=0))/2.
    names = (list(PARAMETERS)+[p+'^2' for p in PARAMETERS]
             +[PARAMETERS[i]+'*'+PARAMETERS[j] for i in range(6) for j in range(i+1, 6)])
    return [dict(feature=name, normalized_obstruction_minus_measured=float(np.clip(value, -1., 1.)),
                 interpretation='tentative observational association; not causal')
            for name, value in zip(names, contrast[1:])]


def select_informative_points(existing_points, outcomes, bounds, count,
                              pending_points=(), seed=0, responses=('discharge','repose')):
    """Return physical ``points`` and JSON-serializable ``diagnostics``.

    Bounds keys/order: sf11, rf11, CED11, sf12, rf12, CED12. Existing/pending
    must be complete in-bounds physical dictionaries (CSV numeric strings work).
    Raw result rows supply QC through adaptive_outcomes.interpret_response.
    Invalid-parameter outcomes are counted/ignored, not imputed. Reservations
    affect prospective coverage only, never observed physics or response fits.
    Candidate exhaustion raises instead of appending unscored random points.
    Caller owns budget, locking, persistence, target exploitation and solver.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
        raise ValueError('count must be a nonnegative integer')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed, count = int(seed), int(count)
    low, high = _limits(bounds)
    kinds=tuple(responses)
    if not kinds or len(set(kinds))!=len(kinds) or set(kinds)-{'discharge','repose','rolling'}:
        raise ValueError('invalid configured responses')
    evidence = _evidence(outcomes, (low, high), kinds)
    history = _unique([_position(p, low, high) for p in list(existing_points)+list(pending_points)]
                      +list(evidence['design']))
    # Attempts constrain novelty, not the information in the measured surface.
    # Only QC-eligible joint responses plus explicitly still-pending proposals
    # contribute to the prospective response-design information matrix.
    pending = _unique([_position(p, low, high) for p in pending_points])
    measured = _unique(evidence['measured_points'])
    information_design = _unique(list(measured)+list(pending))
    f = _quadratic_features(information_design)
    inverse, null, rank = _design(f)
    centers = evidence['labeled'][evidence['labels'] == 0]
    if len(centers) > 12:
        centers = centers[np.linspace(0, len(centers)-1, 12, dtype=int)]
    local = _obstruction_candidates(centers)
    pool = _unique(np.vstack((_candidate_pool(np.random.default_rng(seed)), local)))
    feasibility = _feasibility_probability(pool, evidence['labeled'], evidence['labels'])
    response_uncertainty, response_models, prediction_quality = _response_uncertainty(pool, evidence)
    audit = dict(seed=seed, parameter_order=list(PARAMETERS),
                 parameter_bounds={k:[float(low[i]), float(high[i])] for i,k in enumerate(PARAMETERS)},
                 high_cohesion_quota=0, selections=[],
                 design=dict(prediction_accuracy_claim=False, feature_count=28, rank_before=int(rank), rank_after=int(rank), ridge=_RIDGE,
                             measured_joint_points=len(measured), pending_points=len(pending),
                             information_basis='QC-eligible jointly measured responses plus prospective pending proposals; failed attempts excluded'),
                 pool=dict(generator='seeded_latin_hypercube_plus_boundary_and_local', global_lhs=_POOL_SIZE,
                           total=len(pool), local_candidates=len(local), obstruction_centers=len(centers)),
                 evidence_counts=evidence['counts'], response_models=response_models,
                 prediction_quality=prediction_quality,
                 tentative_associations=_associations(evidence), causal_claim=False,
                 association_warning='Observational and confounded; tentative associations are not causal.',
                 feasibility_model='local beta(2,2), bounded [0.15,0.85]; QC physical evidence only')
    phi = _quadratic_features(pool)
    distances = _minimum_distances(pool, history)
    points = []
    for _ in range(count):
        leverage = np.maximum(np.einsum('ij,jk,ik->i', phi, inverse, phi), 0.)
        gain = np.log1p(leverage)
        information = gain/(1+gain)
        rank_gain = np.clip(np.sum((phi@null)**2, axis=1)/np.sum(phi*phi, axis=1), 0., 1.)
        if response_models:
            base = .4*information+.15*rank_gain+.3*distances+.15*response_uncertainty
        else:
            base = .5*information+.2*rank_gain+.3*distances
        scores = np.clip(base*(.5+.5*feasibility), 0., 1.)
        admissible = np.flatnonzero(distances > _MIN_DISTANCE)
        if not len(admissible):
            raise ValueError('candidate pool exhausted; no partial or random-fill batch returned')
        index = int(admissible[np.argmax(scores[admissible])])
        point = {k:float(np.clip(low[j]+pool[index,j]*(high[j]-low[j]), low[j], high[j]))
                 for j,k in enumerate(PARAMETERS)}
        points.append(point)
        v = inverse@phi[index]
        inverse -= np.outer(v,v)/(1+leverage[index])
        f = np.vstack([f, phi[index]])
        _, null, after = _design(f)
        audit['selections'].append(dict(point=dict(point), score=float(scores[index]),
            candidate_max_score=float(scores[admissible].max()), logdet_gain=float(gain[index]),
            rank_before=int(rank), rank_after=int(after),
            minimum_normalized_distance=float(distances[index]),
            components=dict(information=float(information[index]), rank_gain=float(rank_gain[index]),
                            distance=float(distances[index]), feasibility=float(feasibility[index]),
                            response_uncertainty=float(response_uncertainty[index]))))
        rank = after
        distances = np.minimum(distances, _minimum_distances(pool, pool[index:index+1]))
    audit['design']['rank_after'] = int(rank)
    return dict(points=points, diagnostics=audit)
