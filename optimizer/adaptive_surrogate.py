"""Measured-only independent Matérn GPs and auditable Monte Carlo acquisition."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from optimizer.article_optimizer import weighted_normalized_score
from optimizer.adaptive_outcomes import OUTCOME_CODES, VERSION as OUTCOME_VERSION, interpret_response
from optimizer.protocol_eligibility import interpret_for_controller

VERSION = 'adaptive-matern52-mcei-v2'


def measured(row, kind, controller=None):
    return interpret_for_controller(controller, row, kind)['regression_included']


def normalized(controller, points):
    names = controller.parameter_names
    specs = controller.config['parameters']
    low = np.array([specs[n]['min'] for n in names], float)
    span = np.array([specs[n]['max']-specs[n]['min'] for n in names], float)
    try:
        x = np.array([[float(p[n]) for n in names] for p in points], float).reshape(-1, len(names))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid parameters') from exc
    if (not np.isfinite(x).all() or not np.isfinite(span).all() or (span <= 0).any()
            or (x < low).any() or (x > low+span).any()):
        raise ValueError('finite parameters within exact bounds required')
    return (x-low)/span


class ResponseGP:
    """Standardized output, ARD Matérn 5/2; bounded MLE and Cholesky jitter."""
    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        y = np.asarray(y, float)
        if (self.x.ndim != 2 or len(y) < 2 or y.shape != (len(self.x),)
                or not np.isfinite(self.x).all() or not np.isfinite(y).all()):
            raise ValueError('two measured finite samples required')
        self.center, self.scale = float(y.mean()), max(float(y.std()), 1.)
        z = (y-self.center)/self.scale
        noise = 1e-5

        def factor(log_lengths):
            kernel = self.kernel(self.x, self.x, np.exp(log_lengths))
            for jitter in (noise, 1e-4, 1e-3, 1e-2):
                try:
                    return cho_factor(kernel+np.eye(len(y))*jitter, lower=True), jitter
                except np.linalg.LinAlgError:
                    continue
            raise ValueError('GP factorization failed closed')

        def objective(theta):
            cf, _ = factor(theta)
            return float(.5*z@cho_solve(cf, z)+np.log(np.diag(cf[0])).sum()+len(y)*.5*np.log(2*np.pi))

        initial = np.full(self.x.shape[1], np.log(.3))
        fit = minimize(objective, initial, method='L-BFGS-B',
                       bounds=[(np.log(.03), np.log(3.))]*len(initial),
                       options={'maxiter': 60, 'maxfun': 600})
        # A bounded, finite initial fit remains valid if optimization did not improve it.
        theta = fit.x if np.isfinite(fit.fun) and fit.fun <= objective(initial) else initial
        self.lengths = np.clip(np.exp(theta), .03, 3.)
        self.cf, jitter = factor(np.log(self.lengths))
        self.alpha = cho_solve(self.cf, z)
        self.diagnostics = dict(lengthscales=self.lengths.tolist(), noise_floor=noise, jitter=jitter,
                                optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                                negative_log_likelihood=objective(np.log(self.lengths)),
                                output_center=self.center, output_scale=self.scale)

    @staticmethod
    def kernel(a, b, lengths):
        r = np.sqrt(5.)*cdist(a/lengths, b/lengths)
        return (1+r+r*r/3)*np.exp(-r)

    def predict(self, x):
        k = self.kernel(np.asarray(x, float), self.x, self.lengths)
        v = solve_triangular(self.cf[0], k.T, lower=True)
        mean = self.center+self.scale*(k@self.alpha)
        std = self.scale*np.sqrt(np.maximum(1.-np.sum(v*v, axis=0), 1e-12))
        return mean, std


def propose_next(controller, completed_rows, attempted_points, proposal_index=0):
    """Return one bounded point and JSON diagnostics. All attempts include pending."""
    config = controller.config
    kinds = config['tests']
    outcomes = [[interpret_for_controller(controller, r, k) for k in kinds] for r in completed_rows]
    x = normalized(controller, completed_rows)
    attempts = normalized(controller, list(completed_rows)+list(attempted_points))
    models, training_ids, assigned = {}, {}, {}
    for k in kinds:
        indices = [i for i, result in enumerate(outcomes) if result[kinds.index(k)]['regression_included']]
        try:
            y = np.array([float(completed_rows[i][k+'_deg']) for i in indices])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid measured response') from exc
        if not np.isfinite(y).all():
            raise ValueError('nonfinite measured response')
        if len(indices) < 2:
            raise ValueError('two measured samples required for '+k)
        models[k] = ResponseGP(x[indices], y)
        training_ids[k] = [int(completed_rows[i]['trial_id']) for i in indices]
        assigned[k] = sum(r.get(k+'_response_status') == 'user_assigned_90' for r in completed_rows)
    targets = np.array([config['target_'+k+'_deg'] for k in kinds], float)
    scales = np.array([controller.response_scales[k] for k in kinds], float)
    weights = np.array([config['weights'][k] for k in kinds], float)
    weighted_normalized_score(dict.fromkeys(kinds, 0.), controller.response_scales, config['weights'])

    def score(r):
        try:
            value = np.array([float(r[k+'_deg']) for k in kinds])
            if all(interpret_for_controller(controller, r, k)['regression_included'] for k in kinds) and np.isfinite(value).all():
                return float(np.sum(weights*np.abs(value-targets)/scales)/weights.sum())
        except (KeyError, TypeError, ValueError):
            pass
        return float('inf')

    scores = [score(r) for r in completed_rows]
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
    best = sorted((i for i, r in enumerate(completed_rows)
                   if np.isfinite(scores[i]) and all(measured(r, k) for k in kinds)), key=lambda i: scores[i])[:5]
    local = count//2 if best else 0
    if local:
        centers = x[rng.choice(best, size=local)]
        pool[:local] = np.clip(centers+rng.normal(0, .12, centers.shape), 0., 1.)
    distances = cdist(pool, attempts).min(axis=1) if len(attempts) else np.full(count, np.sqrt(x.shape[1]))
    mask = distances >= distance
    pool, distances = pool[mask], distances[mask]
    if not len(pool):
        raise ValueError('candidate pool exhausted by minimum distance')
    # Outcome smoothing is a heuristic ranking penalty, NOT calibrated probability.
    known = np.array([[float(o['physical_classifier_label'] is not None) for o in result] for result in outcomes])
    labels = np.array([[float(o['physical_classifier_label'] or 0) for o in result] for result in outcomes])
    normals = rng.standard_normal((samples, len(kinds)))
    winner, winning = None, -1.
    for start in range(0, len(pool), 256):
        batch = pool[start:start+256]
        predictions = [models[k].predict(batch) for k in kinds]
        means, stds = np.array([p[0] for p in predictions]).T, np.array([p[1] for p in predictions]).T
        draws = means[:, None, :]+stds[:, None, :]*normals[None, :, :]
        predicted_scores = (np.abs(draws-targets)/scales*weights).sum(axis=2)/weights.sum()
        ei = np.maximum(incumbent-predicted_scores, 0.).mean(axis=1)
        neighbors = np.exp(-cdist(batch, x)**2/(2*.2**2))
        feasibility = ((1.+neighbors@labels)/(2.+neighbors@known)).prod(axis=1)
        acquisition = ei*feasibility
        i = int(np.argmax(acquisition))
        if acquisition[i] > winning:
            winning = float(acquisition[i])
            winner = (start+i, means[i], stds[i], float(predicted_scores[i].mean()), float(ei[i]), float(feasibility[i]))
    index, means, stds, score_mean, ei, feasibility = winner
    specs = config['parameters']
    point = {n: float(np.clip(specs[n]['min']+pool[index, j]*(specs[n]['max']-specs[n]['min']),
                              specs[n]['min'], specs[n]['max'])) for j, n in enumerate(controller.parameter_names)}
    diagnostics = dict(kind='independent-response-gp', version=VERSION, seed=seed,
                       outcome_interpretation_version=OUTCOME_VERSION,
                       outcome_counts={k: {code: sum(result[j]['outcome_code'] == code for result in outcomes)
                                           for code in OUTCOME_CODES} for j, k in enumerate(kinds)},
                       training_ids=training_ids, training_counts={k: len(v) for k, v in training_ids.items()},
                       assigned_count=sum(assigned.values()), assigned_counts=assigned,
                       predicted_means=dict(zip(kinds, means.tolist())), predicted_stddev=dict(zip(kinds, stds.tolist())),
                       predicted_score_mean=score_mean, acquisition=winning, expected_improvement=ei,
                       incumbent_score=incumbent, min_distance=float(distances[index]), required_min_distance=distance,
                       pool=dict(requested=count, eligible=len(pool), local=local, local_std_normalized=.12, mc_samples=samples),
                       feasibility=dict(kind='kernel-smoothed-known-physical-labels-beta11-heuristic', value=feasibility,
                                        known_label_counts={k: int(known[:, j].sum()) for j, k in enumerate(kinds)},
                                        excluded_technical_failures=sum(all(o['outcome_code'] == 'technical_failure'
                                                                           for o in result) for result in outcomes),
                                        physical_diagnosis_verified=False, calibrated_probability=False),
                       fit_diagnostics={k: m.diagnostics for k, m in models.items()})
    return point, diagnostics
