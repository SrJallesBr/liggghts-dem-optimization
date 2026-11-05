"""Auditable MOP-inspired regression selection, NOT a vendor implementation.

NumPy/SciPy only. Local MOP-inspired model-family selection, not ANSYS replication
or global automatic variable-subset optimization. All active dimensions remain
in the acquisition; every family tests full inputs plus an optional predefined
top-2 screening variant. Four candidate types (linear, ridge quadratic, Gaussian MLS,
fixed-kernel Matérn GP) share exact-point grouped CV splits. Input screening is
recomputed strictly within each training fold. CV CoP is a *selection* metric,
not independent/nested validation. Repeats contribute individual PRESS terms.
Uncertainty and physical feasibility are uncalibrated ranking heuristics.
No simulation, disk writes, response assignment, or controller changes occur.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from threading import RLock

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist

from optimizer.adaptive_outcomes import OUTCOME_CODES, VERSION as OUTCOME_VERSION, interpret_response
from optimizer.protocol_eligibility import interpret_for_controller
from optimizer.adaptive_surrogate import normalized, ResponseGP
from optimizer.article_optimizer import weighted_normalized_score

VERSION = 'mop-inspired-groupcv-v1'
METHODOLOGY_SCOPE = ('Local MOP-inspired model-family selection with full inputs and optional '
                     'predefined top-2 screening; not ANSYS replication or global automatic subset search.')
FAMILIES = ('linear', 'ridge_quadratic', 'gaussian_mls_linear', 'matern52_gp')
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_SIZE = 2


def clear_cache():
    """Discard in-memory completed-data fits (useful for offline tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def grouped_folds(x, n_splits=5):
    """Stable exact normalized point groups; no rounding or response access."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite nonempty parameter matrix required')
    _, groups = np.unique(x, axis=0, return_inverse=True)
    count = int(groups.max())+1
    splits = min(int(n_splits), count)
    if splits < 1:
        raise ValueError('positive split count required')
    order = np.random.default_rng(1729).permutation(count)
    assignment = np.empty(count, dtype=int)
    assignment[order] = np.arange(count) % splits
    return groups, assignment[groups]


def _screen(x, y, count):
    """Rank linear OR centered-quadratic marginal association, train data only."""
    if count >= x.shape[1]:
        return np.arange(x.shape[1])
    z = y-y.mean()
    scores = np.zeros(x.shape[1])
    for values in (x, (x-.5)**2):
        centered = values-values.mean(axis=0)
        denominator = np.linalg.norm(centered, axis=0)*np.linalg.norm(z)
        association = np.divide(np.abs(centered.T@z), denominator,
                                out=np.zeros_like(scores), where=denominator > 1e-15)
        scores = np.maximum(scores, association)
    # Stable tie-breaking is by declared parameter order, never held-out values.
    return np.argsort(-scores, kind='stable')[:count]


def _design(x, quadratic=False):
    centered = np.asarray(x, float)-.5
    columns = [np.ones(len(centered)), *centered.T]
    if quadratic:
        columns.extend(centered[:, i]*centered[:, j]
                       for i in range(centered.shape[1]) for j in range(i, centered.shape[1]))
    return np.column_stack(columns)


class _Estimator:
    """Small deterministic regressors; all tunings fixed before observing CV."""
    def __init__(self, family, x, y):
        self.family, self.x = family, np.asarray(x, float)
        self.center = float(y.mean())
        self.scale = max(float(y.std()), 1.)
        self.z = (y-self.center)/self.scale
        self.bandwidth = .35
        self.lengthscale = .5
        self.noise = 1e-5
        if family in ('linear', 'ridge_quadratic'):
            design = _design(self.x, family == 'ridge_quadratic')
            if family == 'linear':
                self.beta = np.linalg.lstsq(design, self.z, rcond=1e-10)[0]
            else:
                penalty = np.eye(design.shape[1])*1e-6
                penalty[0, 0] = 0.
                self.beta = np.linalg.solve(design.T@design+penalty, design.T@self.z)
        elif family == 'matern52_gp':
            self.lengths = np.full(self.x.shape[1], self.lengthscale)
            k = ResponseGP.kernel(self.x, self.x, self.lengths)
            self.cf = cho_factor(k+np.eye(len(y))*self.noise, lower=True)
            self.alpha = cho_solve(self.cf, self.z)
        elif family != 'gaussian_mls_linear':
            raise ValueError('unknown regression family')

    def predict(self, x):
        x = np.asarray(x, float)
        if self.family in ('linear', 'ridge_quadratic'):
            mean = _design(x, self.family == 'ridge_quadratic')@self.beta
            std = np.zeros(len(x))
        elif self.family == 'matern52_gp':
            kernel = ResponseGP.kernel(x, self.x, self.lengths)
            v = solve_triangular(self.cf[0], kernel.T, lower=True)
            mean = kernel@self.alpha
            std = np.sqrt(np.maximum(1.-np.square(v).sum(axis=0), 0.))*self.scale
        else:
            # Gaussian moving least squares: centered local linear model with
            # a tiny slope ridge; local intercept is unpenalized. Chunk at caller.
            delta = self.x[None, :, :]-x[:, None, :]
            w = np.exp(-np.square(delta).sum(axis=2)/(2*self.bandwidth**2))
            w /= np.maximum(w.max(axis=1, keepdims=True), 1e-300)
            design = np.concatenate((np.ones((*delta.shape[:2], 1)), delta), axis=2)
            gram = np.einsum('bni,bn,bnj->bij', design, w, design, optimize=True)
            rhs = np.einsum('bni,bn,n->bi', design, w, self.z, optimize=True)
            ridge = np.eye(design.shape[-1])*1e-6
            ridge[0, 0] = 0.
            mean = np.linalg.solve(gram+ridge, rhs[..., None])[..., 0][:, 0]
            std = np.zeros(len(x))
        return self.center+self.scale*mean, std


class ResponseMOP:
    """Select a response regressor and subset with grouped five-fold CV CoP.

    Fixed kernel hyperparameters avoid costly repeated GP MLE and are exposed
    explicitly. Full inputs and train-fold-ranked top-2 inputs are tested for
    every family. No CV is claimed for a single group/single available fold.
    """
    def __init__(self, x, y, parameter_names=None, *, fold_ids=None, group_ids=None,
                 trial_ids=None, row_indices=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if (x.ndim != 2 or not x.shape[1] or len(y) < 2 or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or (x < 0).any() or (x > 1).any()):
            raise ValueError('two measured finite samples with normalized parameters required')
        names = tuple(parameter_names) if parameter_names is not None else tuple('x'+str(i) for i in range(x.shape[1]))
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError('parameter names must match dimensions')
        local_groups, default_folds = grouped_folds(x)
        groups = local_groups if group_ids is None else np.asarray(group_ids, int)
        folds = default_folds if fold_ids is None else np.asarray(fold_ids, int)
        ids = list(range(len(x))) if trial_ids is None else [int(i) for i in trial_ids]
        indices = list(range(len(x))) if row_indices is None else [int(i) for i in row_indices]
        if groups.shape != (len(x),) or folds.shape != (len(x),) or len(ids) != len(x) or len(indices) != len(x):
            raise ValueError('group/fold/sample metadata must match samples')
        for group in np.unique(local_groups):
            if len(np.unique(folds[local_groups == group])) != 1 or len(np.unique(groups[local_groups == group])) != 1:
                raise ValueError('repeated parameter point cannot cross folds/groups')
        for group in np.unique(groups):
            if len(np.unique(folds[groups == group])) != 1:
                raise ValueError('group cannot cross folds')
        subsets = [('full', x.shape[1])]
        if x.shape[1] > 2:
            subsets.append(('ranked_top2', 2))
        available = len(np.unique(folds)) >= 2
        sst = float(np.square(y-y.mean()).sum())
        zero_variance = sst <= np.finfo(float).eps*max(1., float(y@y))
        variants = []
        for family in FAMILIES:
            for subset, width in subsets:
                record = dict(model=family, inputs=subset, input_count=width,
                              status='ok' if available else 'insufficient_distinct_folds',
                              folds=[], cop=None, cop_status='unavailable', cv_rmse_deg=None,
                              cv_mae_deg=None, heldout_errors_deg=[],
                              press_deg2=None, sst_deg2=sst, oof_trial_ids=[], oof_predictions_deg=[])
                if available:
                    prediction = np.empty(len(y))
                    for fold in np.unique(folds):
                        test, train = folds == fold, folds != fold
                        selected = _screen(x[train], y[train], width)
                        estimator = _Estimator(family, x[train][:, selected], y[train])
                        mean, _ = estimator.predict(x[test][:, selected])
                        if not np.isfinite(mean).all():
                            raise ValueError('nonfinite cross-validation prediction')
                        prediction[test] = mean
                        record['folds'].append(dict(
                            fold_id=int(fold), selected_inputs=[names[j] for j in selected],
                            train_group_ids=np.unique(groups[train]).tolist(),
                            test_group_ids=np.unique(groups[test]).tolist(),
                            train_row_indices=[indices[j] for j in np.flatnonzero(train)],
                            test_row_indices=[indices[j] for j in np.flatnonzero(test)],
                            train_trial_ids=[ids[j] for j in np.flatnonzero(train)],
                            test_trial_ids=[ids[j] for j in np.flatnonzero(test)],
                            predictions_deg=mean.tolist()))
                    press = float(np.square(y-prediction).sum())
                    record.update(press_deg2=press, cv_rmse_deg=float(np.sqrt(press/len(y))),
                                  cv_mae_deg=float(np.abs(prediction-y).mean()),
                                  heldout_errors_deg=(prediction-y).tolist(),
                                  cop=None if zero_variance else float(1.-press/sst),
                                  cop_status='undefined_zero_variance' if zero_variance else 'defined',
                                  oof_trial_ids=ids, oof_predictions_deg=prediction.tolist())
                variants.append(record)
        # All candidates have identical observed rows and folds: minimizing PRESS
        # is equivalent to maximizing CoP. For constant outputs use RMSE without
        # fabricating a CoP; for no CV use first (linear/full), explicitly flagged.
        best = min(variants, key=lambda r: r['press_deg2']) if available else variants[0]
        self.selected = _screen(x, y, best['input_count'])
        self.estimator = _Estimator(best['model'], x[:, self.selected], y)
        self.x = x[:, self.selected]
        self.input_dim = x.shape[1]
        self.cv_error = best['cv_rmse_deg'] if available else max(float(y.std()), 1.)
        self.diagnostics = dict(
            selected_model=best['model'], selected_input_variant=best['inputs'],
            selected_inputs=[names[i] for i in self.selected],
            training_count=len(y), training_angles_deg=y.tolist(), training_ids=ids,
            cv_available=available, cv_folds=len(np.unique(folds)), cop=best['cop'],
            cop_status=best['cop_status'], cv_rmse_deg=best['cv_rmse_deg'],
            tested_variants=variants, metric='CoP = 1 - PRESS / SST; row-weighted grouped OOF',
            independent_validation=False,
            validation_limitation='CV reused for model selection; not independent or nested validation.',
            uncertainty_calibrated=False, vendor_implementation=False,
            methodology_scope=METHODOLOGY_SCOPE,
            screening='training-fold-only max absolute linear/quadratic marginal correlation; stable ties',
            fixed_hyperparameters=dict(quadratic_ridge=1e-6, mls_bandwidth_normalized=.35,
                                       mls_slope_ridge=1e-6, gp_lengthscale_normalized=.5, gp_noise=1e-5),
            gp_hyperparameters_optimized=False,
            uncertainty='GP latent std plus CV residual floor; other models CV residual/distance heuristic',
            no_cv_fallback=None if available else 'linear/full, no validation claim')

    def predict(self, x):
        result = self.predict_components(x)
        return result['mean'], result['stddev']

    def predict_components(self, x):
        """Separate GP latent posterior std from the CV residual/distance proxy.

        GP variance is conditional on fixed kernel/noise and ignores model-selection
        uncertainty. The residual floor is a heuristic, NOT a calibrated confidence
        interval or predictive distribution, for GP and non-GP families alike.
        """
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError('finite prediction parameters with matching dimensions required')
        means, stds, native_stds, proxies = [], [], [], []
        for start in range(0, len(x), 256):
            selected = x[start:start+256, self.selected]
            mean, native_std = self.estimator.predict(selected)
            nearest = cdist(selected, self.x).min(axis=1)
            floor = max(float(self.cv_error), .05)*(1.+nearest)
            std = np.sqrt(np.square(native_std)+np.square(floor))
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError('nonfinite surrogate prediction')
            means.extend(mean.tolist())
            stds.extend(std.tolist())
            native_stds.extend(native_std.tolist())
            proxies.extend(floor.tolist())
        return dict(mean=np.asarray(means), stddev=np.asarray(stds),
                    gp_latent_stddev=np.asarray(native_stds),
                    cv_residual_distance_proxy=np.asarray(proxies))


def _data(controller, rows):
    x = normalized(controller, rows)
    kinds = controller.config['tests']
    outcomes = [[interpret_for_controller(controller, row, k) for k in kinds] for row in rows]
    return x, outcomes


def fit_responses(controller, completed_rows):
    """Uncached fitting API, no mutation; each assay uses only measured angles."""
    rows = list(completed_rows)
    kinds, names = controller.config['tests'], controller.parameter_names
    x, outcomes = _data(controller, rows)
    if not len(x):
        raise ValueError('two measured samples required for every response')
    groups, folds = grouped_folds(x)
    models, training_ids, assigned = {}, {}, {}
    for j, k in enumerate(kinds):
        valid = [i for i, result in enumerate(outcomes) if result[j]['regression_included']]
        if len(valid) < 2:
            raise ValueError('two measured samples required for '+k)
        # InterpretResponse handles flat OR nested rows; never read adopted 90s
        # directly and never fill missing assays from another response.
        angles = np.asarray([outcomes[i][j]['angle_deg'] for i in valid], float)
        ids = [int(rows[i]['trial_id']) for i in valid]
        models[k] = ResponseMOP(x[valid], angles, names, fold_ids=folds[valid],
                                group_ids=groups[valid], trial_ids=ids, row_indices=valid)
        training_ids[k] = ids
        assigned[k] = sum(result[j]['legacy_response_status'] == 'user_assigned_90' for result in outcomes)
    diagnostics = dict(
        training_ids=training_ids, training_counts={k: len(v) for k, v in training_ids.items()},
        assigned_counts=assigned, assigned_count=sum(assigned.values()),
        split_policy='exact normalized parameter groups; shared response-blind seeded folds',
        split_seed=1729,
        split_groups=[dict(row_index=i, trial_id=int(rows[i]['trial_id']), group_id=int(groups[i]),
                           fold_id=int(folds[i]), normalized_point=x[i].tolist()) for i in range(len(rows))],
        fit_diagnostics={k: model.diagnostics for k, model in models.items()})
    return models, diagnostics


def _cached_fit(controller, rows, x, outcomes):
    # Fingerprint exactly the inputs used by regression/diagnostics, not pending
    # proposals. Invalid raw numbers are represented by interpreted None, not NaN.
    payload = dict(version=VERSION, outcome_version=OUTCOME_VERSION, names=list(controller.parameter_names),
                   parameters=controller.config['parameters'], kinds=controller.config['tests'],
                   points=x.tolist(), ids=[int(r['trial_id']) for r in rows],
                   outcomes=[[[o['angle_deg'], o['outcome_code'], o['regression_included'], o['physical_classifier_label'],
                               o['legacy_response_status']] for o in result] for result in outcomes])
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf8')).hexdigest()
    # Serialize first fit of a signature; concurrent pending proposals reuse it.
    with _CACHE_LOCK:
        if signature not in _CACHE:
            _CACHE[signature] = fit_responses(controller, rows)
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(signature)
        models, diagnostics = _CACHE[signature]
        return models, deepcopy(diagnostics), signature


def propose_next(controller, completed_rows, attempted_points, proposal_index=0):
    """Adaptive-refinement proposer: bounded pool, measured score, heuristic MC EI.

    Mirrors adaptive_surrogate acquisition without global monkeypatching or model
    refits for pending proposals. All completed and attempted/pending points are
    excluded by Euclidean normalized distance >= configured threshold (>= .01).
    """
    rows = list(completed_rows)
    config, kinds = controller.config, controller.config['tests']
    x, outcomes = _data(controller, rows)
    attempts = normalized(controller, rows+list(attempted_points))
    models, diagnostics, signature = _cached_fit(controller, rows, x, outcomes)
    weighted_normalized_score(dict.fromkeys(kinds, 0.), controller.response_scales, config['weights'])
    targets = np.array([config['target_'+k+'_deg'] for k in kinds], float)
    scales = np.array([controller.response_scales[k] for k in kinds], float)
    weights = np.array([config['weights'][k] for k in kinds], float)
    if not np.isfinite(targets).all():
        raise ValueError('finite targets required')
    scores = []
    for result in outcomes:
        value = (float(np.sum(weights*np.abs(np.array([o['angle_deg'] for o in result])-targets)/scales)/weights.sum())
                 if all(o['regression_included'] for o in result) else float('inf'))
        scores.append(value)
    incumbent = min(scores, default=float('inf'))
    if not np.isfinite(incumbent):
        raise ValueError('at least one fully measured completed incumbent required')
    count = int(config.get('model_candidate_pool', 6000))
    samples = int(config.get('adaptive_mc_samples', 128))
    distance = float(config.get('adaptive_min_distance', .01))
    if not 2 <= count <= 20000 or not 16 <= samples <= 512 or not np.isfinite(distance) or distance < .01:
        raise ValueError('invalid bounded pool/Monte Carlo count or minimum distance (<0.01)')
    seed = int(config.get('random_seed', 1))+int(proposal_index)*104729
    rng = np.random.default_rng(seed)
    pool = rng.random((count, x.shape[1]))
    best = sorted((i for i in range(len(rows)) if np.isfinite(scores[i])), key=lambda i: scores[i])[:5]
    local = count//2
    centers = x[rng.choice(best, size=local)]
    pool[:local] = np.clip(centers+rng.normal(0, .12, centers.shape), 0., 1.)
    # Round trip into physical bounds before distance checking to honor exact
    # represented parameters, including large CED and nonzero lower bounds.
    specs = config['parameters']
    low = np.array([specs[n]['min'] for n in controller.parameter_names], float)
    span = np.array([specs[n]['max']-specs[n]['min'] for n in controller.parameter_names], float)
    physical = np.clip(low+pool*span, low, low+span)
    pool = (physical-low)/span
    distances = cdist(pool, attempts).min(axis=1)
    mask = distances >= distance
    pool, physical, distances = pool[mask], physical[mask], distances[mask]
    if not len(pool):
        raise ValueError('candidate pool exhausted by minimum distance')
    known = np.array([[float(o['physical_classifier_label'] is not None) for o in result] for result in outcomes])
    labels = np.array([[float(o['physical_classifier_label'] or 0) for o in result] for result in outcomes])
    normals = rng.standard_normal((samples, len(kinds)))
    winner, winning = None, -1.
    for start in range(0, len(pool), 256):
        batch = pool[start:start+256]
        prediction = [models[k].predict_components(batch) for k in kinds]
        means = np.array([p['mean'] for p in prediction]).T
        stds = np.array([p['stddev'] for p in prediction]).T
        draws = means[:, None, :]+stds[:, None, :]*normals[None, :, :]
        predicted_scores = (np.abs(draws-targets)/scales*weights).sum(axis=2)/weights.sum()
        ei = np.maximum(incumbent-predicted_scores, 0.).mean(axis=1)
        neighbors = np.exp(-cdist(batch, x)**2/(2*.2**2))
        feasibility = ((1.+neighbors@labels)/(2.+neighbors@known)).prod(axis=1)
        acquisition = ei*feasibility
        i = int(np.argmax(acquisition))
        if acquisition[i] > winning:
            winning = float(acquisition[i])
            components = {k: dict(gp_latent_stddev_deg=float(prediction[j]['gp_latent_stddev'][i]),
                                  cv_residual_distance_proxy_deg=float(prediction[j]['cv_residual_distance_proxy'][i]),
                                  combined_stddev_deg=float(stds[i, j]), calibrated=False)
                          for j, k in enumerate(kinds)}
            winner = (start+i, means[i], stds[i], float(predicted_scores[i].mean()), float(ei[i]), float(feasibility[i]), components)
    index, means, stds, score_mean, ei, feasibility, components = winner
    point = {n: float(physical[index, j]) for j, n in enumerate(controller.parameter_names)}
    diagnostics.update(
        kind='mop-inspired-independent-response-selection', version=VERSION, seed=seed,
        methodology_scope=METHODOLOGY_SCOPE,
        completed_data_signature=signature, vendor_implementation=False, uncertainty_calibrated=False,
        independent_validation=False,
        validation_limitation='CV reused for model selection; not independent or nested validation.',
        outcome_interpretation_version=OUTCOME_VERSION,
        outcome_counts={k: {code: sum(result[j]['outcome_code'] == code for result in outcomes)
                            for code in OUTCOME_CODES} for j, k in enumerate(kinds)},
        predicted_means=dict(zip(kinds, means.tolist())), predicted_stddev=dict(zip(kinds, stds.tolist())),
        uncertainty_components=components,
        predicted_score_mean=score_mean, acquisition=winning, expected_improvement=ei,
        incumbent_score=incumbent, min_distance=float(distances[index]), required_min_distance=distance,
        pool=dict(requested=count, eligible=len(pool), local=local, local_std_normalized=.12, mc_samples=samples),
        feasibility=dict(kind='kernel-smoothed-known-physical-labels-beta11-heuristic', value=feasibility,
                         known_label_counts={k: int(known[:, j].sum()) for j, k in enumerate(kinds)},
                         excluded_technical_failures=sum(all(o['outcome_code'] == 'technical_failure'
                                                            for o in result) for result in outcomes),
                         physical_diagnosis_verified=False, calibrated_probability=False))
    return point, diagnostics
