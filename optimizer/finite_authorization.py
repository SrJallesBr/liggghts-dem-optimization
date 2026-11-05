"""Separate finite admission; the manifest digest is the local trust anchor."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from runtime.continuation_migration import BUDGET, CAMPAIGN, SCOPE

ROOT = Path('/mnt/c/DEM/campanhas/fresh_doe84_dt100us_20261104')
MANIFEST_SHA256 = '7e4b0a8eb1592503573bebdf0f7e9e30b9694520bbd7114752283bcecf530d3f'
_active = False


def artifact_hash(path, rel):
    data = path.read_bytes()
    if rel == 'optimizer/finite_authorization.py':
        data, count = re.subn(rb"(?m)^MANIFEST_SHA256 = '[0-9a-f]{64}'(?=\r?$)", b"MANIFEST_SHA256 = '<manifest-digest>'", data)
        if count != 1:
            raise RuntimeError('HOLD: invalid authorization trust anchor')
    return hashlib.sha256(data).hexdigest()


def require_finite_authorization(root, config):
    from optimizer.calibration_controller import load_configuration
    root = Path(root).resolve()
    if not _active or root != ROOT.resolve() or os.environ.get('DEM_FINITE_AUTHORIZATION_ID') != SCOPE:
        raise RuntimeError('HOLD: finite authorization or launcher context missing')
    path = root / 'finite_authorization.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError('HOLD: finite authorization manifest mismatch')
    approval = json.loads(path.read_text(encoding='utf8'))
    if approval.get('scope') != SCOPE or approval.get('root') != str(ROOT):
        raise RuntimeError('HOLD: finite scope mismatch')
    for rel, expected in approval['sha256'].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file() or artifact_hash(p, rel) != expected:
            raise RuntimeError('HOLD: finite pinned artifact mismatch: ' + rel)
    expected_config = copy.deepcopy(load_configuration(root / 'configuracao_otimizacao.txt'))
    expected_config.update(BUDGET)
    expected_config['campaign'].update(CAMPAIGN)
    if config != expected_config:
        raise RuntimeError('HOLD: finite configuration mismatch')
    return approval
