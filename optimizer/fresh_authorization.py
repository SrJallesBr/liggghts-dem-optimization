"""Portable, explicitly opt-in admission for a fresh independent finite package."""
import hashlib
import json
import os
import re
from pathlib import Path

SCOPE = 'fresh-independent-finite-doe84-retry420-cap1000-20-v1'
MANIFEST_SHA256 = '3d5bef7b7c0569bcca1495d3be631fb79d652ff12a081df636cd24aa18f74384'


def artifact_hash(path, rel):
    data=Path(path).read_bytes()
    if rel=='optimizer/fresh_authorization.py':
        data,n=re.subn(rb"(?m)^MANIFEST_SHA256 = '[0-9a-f]{64}'(?=\r?$)",b"MANIFEST_SHA256 = '<manifest-digest>'",data)
        if n!=1:
            raise RuntimeError('HOLD: invalid manifest trust anchor')
    return hashlib.sha256(data).hexdigest()


def require_fresh_authorization(root, config):
    root=Path(root).resolve()
    if os.environ.get('DEM_FRESH_FINITE_AUTHORIZATION_ID') != SCOPE:
        raise RuntimeError('HOLD: explicit fresh finite launch authorization missing')
    manifest=root/'fresh_finite_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=MANIFEST_SHA256:
        raise RuntimeError('HOLD: fresh finite manifest mismatch')
    approval=json.loads(manifest.read_text())
    if approval.get('scope')!=SCOPE:
        raise RuntimeError('HOLD: fresh finite scope mismatch')
    for rel,digest in approval['sha256'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or not p.is_file() or artifact_hash(p,rel)!=digest:
            raise RuntimeError('HOLD: pinned artifact mismatch: '+rel)
    from optimizer.calibration_controller import load_configuration
    if config!=load_configuration(root/'configuracao_otimizacao.txt'):
        raise RuntimeError('HOLD: runtime configuration differs from pinned file')
    expected={'maximum_doe_attempts':420,'minimum_doe_complete':84,'doe_attempt_multiplier':5,
              'parallel_trials':20,'max_trials':1000,'refinement_trials':748,'first_trial_id':1}
    if any(config.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('HOLD: fresh finite budget mismatch')
    if config['sequential_experiment'].get('imported_doe_rows')!=0:
        raise RuntimeError('HOLD: historical imports forbidden')
    if config.get('fresh_finite_independent') is not True:
        raise RuntimeError('HOLD: not an independent fresh campaign')
    if config.get('mpi_processes_per_trial') != 1:
        raise RuntimeError('HOLD: expected one rank per trial')
    if config.get('campaign') != {'quadratic_terms':28,'doe_trials':84,'refinement_trials':748,'validation_trials':0,'max_trials':1000}:
        raise RuntimeError('HOLD: campaign budget mismatch')
    from .repose_ansys import verify_overlay
    verify_overlay(root)
    return approval
