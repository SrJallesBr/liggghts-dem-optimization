import random
import math


def quadratic_term_count(parameter_count):
    if parameter_count < 1:
        raise ValueError("at least one parameter is required")
    return (parameter_count + 1) * (parameter_count + 2) // 2


def recommend_campaign_size(parameter_names):
    parameter_count = len(tuple(parameter_names))
    terms = quadratic_term_count(parameter_count)
    doe_trials = 3 * terms
    refinement_trials = max(12, 3 * parameter_count)
    validation_trials = max(8, 2 * parameter_count)
    return {"quadratic_terms": terms, "doe_trials": doe_trials, "refinement_trials": refinement_trials, "validation_trials": validation_trials, "max_trials": doe_trials + refinement_trials + validation_trials}


def latin_hypercube(count, bounds, seed):
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    columns = {}
    for name, (low, high) in bounds.items():
        order = list(range(count))
        rng.shuffle(order)
        columns[name] = [low + (stratum + rng.random()) / count * (high - low) for stratum in order]
    return [{name: columns[name][index] for name in bounds} for index in range(count)]


def weighted_normalized_score(errors, maxima, weights):
    """Eq. 4 adaptation: weighted MEAN using fixed project response scales.

    ``maxima`` is retained as an API name for legacy callers, but callers must
    supply fixed tolerances, never candidate-dependent observed maxima.
    """
    if not errors or errors.keys() != maxima.keys() or errors.keys() != weights.keys():
        raise ValueError('errors, fixed scales and weights require identical response keys')
    for name in errors:
        for value, positive in ((errors[name], False), (maxima[name], True), (weights[name], True)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError('finite nonnegative errors and positive scales/weights required')
    return sum(weights[k] * errors[k] / maxima[k] for k in errors) / sum(weights.values())
