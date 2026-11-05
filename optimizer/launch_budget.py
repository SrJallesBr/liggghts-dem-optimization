"""Launch-only DOE attempt ceiling. Does not change measurement/acceptance."""
def limit_doe_queue(points, phase, start_trial_id, ceiling):
    if phase != 'doe':
        return points
    if start_trial_id > ceiling:
        raise RuntimeError(f'DOE stopped at {ceiling} attempts: insufficient accepted measurements; no forced fit.')
    return points[:ceiling - start_trial_id + 1]
