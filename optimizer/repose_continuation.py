"""Continuation policy only for the latest config/code-bound authorization."""
import json
from pathlib import Path


def authorized_continuation(controller):
    path=Path(controller.project_root)/'launch_authorization.json'
    if not path.exists(): return False
    approval=json.loads(path.read_text())
    if approval.get('scope') not in ('authorized-sequential-repose10-provenance-v1', 'authorized-sequential-dt100us-v1'): return False
    from .launch_authorization import require_launch_authorization
    require_launch_authorization(controller.project_root,controller.config)
    return True


def reconcile_completed_batch(root,reserved,terminal):
    path=Path(root)/'exact12_batch.json'
    batch=json.loads(path.read_text())
    start,target=batch['start_reservations'],batch['target_reservations']
    first=batch['first_trial_id']
    ids=set(range(first,first+batch['requested_attempts']))
    if target-start!=12 or batch['requested_attempts']!=12 or len(reserved)<target or not ids<=set(terminal):
        raise RuntimeError('prior exact12 batch incomplete; continuation denied')
    if not set(reserved)<=set(terminal):
        raise RuntimeError('orphan reservation; continuation denied')
    return {'prior_batch_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
            'status':'verified_terminal_preserved','trial_ids':sorted(ids)}
