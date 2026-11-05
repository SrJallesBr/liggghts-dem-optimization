"""Finite sequential search around the existing measured-only MOP models.

Every fifth reservation is deterministic bounded maximin exploration, irrespective
of model confidence. No solver launch, measurement assignment, or disk I/O occurs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from optimizer import mop_surrogate as mop
from optimizer.adaptive_surrogate import normalized
from optimizer.article_optimizer import weighted_normalized_score
from optimizer.adaptive_outcomes import OUTCOME_CODES, VERSION as OUTCOME_VERSION

VERSION = 'sequential-maximin-mop-v1'
GLOBAL_EVERY = 5


def _proposal_options(config):
    options = config.get('adaptive_refinement', {})
    if not isinstance(options, dict):
        raise ValueError('adaptive_refinement must be a mapping')
    count = options.get('candidate_pool', config.get('model_candidate_pool', 6000))
    distance = float(options.get('minimum_distance', config.get('adaptive_min_distance', .01)))
    samples = config.get('adaptive_mc_samples', 128)
    if (type(count) is not int or not 8 <= count <= 20000 or
            type(samples) is not int or not 16 <= samples <= 512 or
            not np.isfinite(distance) or not .01 <= distance <= np.sqrt(len(config['parameters']))):
        raise ValueError('invalid bounded candidate pool/Monte Carlo count or minimum distance (<0.01)')
    return count, distance, samples


def propose_next(controller, completed_rows, attempted_points, proposal_index=0):
    """Proposer with the scheduler's existing reservation-index interface."""
    if type(proposal_index) is not int or proposal_index < 0:
        raise ValueError('nonnegative integer proposal index required')
    rows, attempted_points = list(completed_rows), list(attempted_points)
    count, distance, _ = _proposal_options(controller.config)
    from optimizer.protocol_eligibility import overlay_config,interpret_for_controller
    schema2=hasattr(controller,'project_root') and (overlay_config(controller.project_root) or {}).get('schema_version')==2
    support={k:sum(interpret_for_controller(controller,r,k)['regression_included'] for r in rows) for k in controller.config['tests']} if schema2 else {}
    joint=any(all(interpret_for_controller(controller,r,k)['regression_included'] for k in controller.config['tests']) for r in rows) if schema2 else True
    bootstrap=schema2 and (not joint or any(n<2 for n in support.values()))
    if not bootstrap and (proposal_index+1) % GLOBAL_EVERY:
        completed = sum(r.get('phase') == 'refinement' and r.get('status') in ('complete', 'failed') for r in rows)
        budget = controller.config.get('sequential_experiment', {}).get(
            'new_attempt_budget', controller.config.get('campaign', {}).get('refinement_trials', 48))
        if type(budget) is not int or budget <= 0:
            raise ValueError('positive integer sequential new attempt budget required for proposal')
        radius = .12-(.12-.03)*min(completed/budget, 1.)
        point, diagnostics = _propose_mop(controller, rows, attempted_points, proposal_index, radius)
        diagnostics.update(strategy='mop_adaptive_local', sequential_version=VERSION,
                           proposal_index=proposal_index, global_every=GLOBAL_EVERY,
                           completed_new_attempts=completed, progress_fraction=min(completed/budget, 1.))
        return point, diagnostics
    attempts = attempted_points
    config = controller.config
    existing = normalized(controller, rows+attempts)
    seed = int(config.get('random_seed', 1))+proposal_index*104729
    pool = np.random.default_rng(seed).random((count, len(controller.parameter_names)))
    specs = config['parameters']
    low = np.array([specs[n]['min'] for n in controller.parameter_names], float)
    span = np.array([specs[n]['max']-specs[n]['min'] for n in controller.parameter_names], float)
    physical = np.clip(low+pool*span, low, low+span)
    pool = (physical-low)/span
    distances = cdist(pool, existing).min(axis=1) if len(existing) else np.ones(count)*np.sqrt(pool.shape[1])
    eligible = distances >= distance
    if not eligible.any():
        raise ValueError('candidate pool exhausted by minimum distance')
    index = int(np.argmax(np.where(eligible, distances, -1.)))
    point = {n: float(physical[index, j]) for j, n in enumerate(controller.parameter_names)}
    return point, dict(kind=VERSION, version=VERSION, strategy='protocol_bootstrap_maximin' if bootstrap else 'scheduled_global_maximin',
                       protocol_support=support,
                       proposal_index=proposal_index, global_every=GLOBAL_EVERY, seed=seed,
                       min_distance=float(distances[index]), required_min_distance=distance,
                       acquisition_kind='maximin_normalized_euclidean_distance',
                       pool=dict(requested=count, eligible=int(eligible.sum()), local=0),
                       response_predictions_used=False, convergence_guaranteed=False)


def _propose_mop(controller, completed_rows, attempted_points, proposal_index, radius):
    """MOP acquisition preserved verbatim except the local candidate radius.

    Reuses the original measured-only cache/model fitting. This small pool and
    ranking adapter avoids modifying or monkeypatching mop_surrogate globally.
    Keep acquisition changes in sync with the original proposer.
    """
    rows = list(completed_rows)
    config, kinds = controller.config, controller.config['tests']
    x, outcomes = mop._data(controller, rows)
    attempts = normalized(controller, rows+list(attempted_points))
    models, diagnostics, signature = mop._cached_fit(controller, rows, x, outcomes)
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
    count, distance, samples = _proposal_options(config)
    seed = int(config.get('random_seed', 1))+int(proposal_index)*104729
    rng = np.random.default_rng(seed)
    pool = rng.random((count, x.shape[1]))
    best = sorted((i for i in range(len(rows)) if np.isfinite(scores[i])), key=lambda i: scores[i])[:5]
    local = count//2
    centers = x[rng.choice(best, size=local)]
    pool[:local] = np.clip(centers+rng.normal(0, radius, centers.shape), 0., 1.)
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
        kind='mop-inspired-independent-response-selection', version=mop.VERSION, seed=seed,
        methodology_scope=mop.METHODOLOGY_SCOPE,
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
        pool=dict(requested=count, eligible=len(pool), local=local, local_std_normalized=radius, mc_samples=samples),
        feasibility=dict(kind='kernel-smoothed-known-physical-labels-beta11-heuristic', value=feasibility,
                         known_label_counts={k: int(known[:, j].sum()) for j, k in enumerate(kinds)},
                         excluded_technical_failures=sum(all(o['outcome_code'] == 'technical_failure'
                                                            for o in result) for result in outcomes),
                         physical_diagnosis_verified=False, calibrated_probability=False))
    return point, diagnostics

