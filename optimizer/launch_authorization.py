"""Admission for the explicitly authorized, isolated initial 84-case campaign."""
import hashlib
import json
import os
from pathlib import Path


def require_launch_authorization(root, config):
    root = Path(root).resolve()
    if config.get('fresh_finite_independent') is True:
        from optimizer.fresh_authorization import require_fresh_authorization
        return require_fresh_authorization(root, config)
    if config.get('doe_attempt_multiplier') == 5:
        from optimizer.finite_authorization import require_finite_authorization
        return require_finite_authorization(root, config)
    prefix = 'HOLD: fresh DOE admission failed: '
    if os.environ.get('DEM_NEW_USER_AUTHORIZATION_ID') != 'fresh-doe84-dt100us-v1':
        raise RuntimeError(prefix + 'authorization token missing')
    approval = json.loads((root / 'launch_authorization.json').read_text(encoding='utf-8'))
    if root.name != 'fresh_doe84_dt100us_20261104' or approval.get('scope') != 'fresh-doe84-dt100us-v1':
        raise RuntimeError(prefix + 'campaign scope mismatch')
    if (any(config.get('campaign', {}).get(k) != v for k, v in {'doe_trials': 84, 'refinement_trials': 0, 'validation_trials': 0, 'max_trials': 84}.items())
            or config.get('maximum_doe_attempts') != 84 or config.get('doe_attempt_multiplier') != 1
            or config.get('parallel_trials') != 12 or config.get('first_trial_id') != 593):
        raise RuntimeError(prefix + 'configuration outside authorized scope')
    for rel, expected in approval['sha256'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(prefix + 'pinned artifact mismatch: ' + rel)
    return approval
