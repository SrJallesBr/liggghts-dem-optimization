"""Assay-local admission from pinned inputs and controller-produced final measurement.

Historical rows are untouched. IDs, response labels and revision strings alone
never admit repose. Rolling/discharge retain their independent policies.
"""
import hashlib
import json
import math
import os
from pathlib import Path
from .adaptive_outcomes import interpret_response


def overlay_config(root):
    path = Path(root) / 'runtime' / 'future_repose_overlay.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_repose_snapshot(path):
    path = Path(path)
    case = path.parents[1]
    deck = case/'in.repouso'
    if deck.is_file() and 'ANSYS_REPOSE_10S_ONE_SHOT' in deck.read_text():
        return case/'post/repouso_10s.vtk'
    return path


def _input_pins(controller, workspace, historical=False):
    from .repose_ansys import verify_overlay, REVIEWED_ASSETS
    from .calibration_controller import rewrite_input_parameters, rewrite_input_seeds
    import tempfile
    root=Path(controller.project_root); workspace=Path(workspace).resolve()
    if not workspace.is_relative_to(Path(controller.run_root).resolve()):
        raise ValueError('measurement workspace outside campaign run')
    verify_overlay(root)
    parameters=json.loads((workspace/'parameters.json').read_text())
    point=parameters['parameters']
    if set(point)!=set(controller.parameter_names) or not controller._is_within_active_bounds(point):
        raise ValueError('incompatible parameter provenance')
    # Compare a fresh rewrite of the immutable reviewed template, NOT a label.
    with tempfile.TemporaryDirectory() as td:
        expected=Path(td)/'in.repouso'
        source = root/'Ang Repouso/in.repouso'
        if historical:
            source = root/'provenance/repose11_future_20261104T174349Z/Ang Repouso/in.repouso'
            if digest(source) != '43ff7c774373cd81453c9133218f6faf3815fcf8c25beb1da783f32a3d89d910':
                raise ValueError('historical template pin mismatch')
        expected.write_bytes(source.read_bytes())
        rewrite_input_parameters(expected,point)
        rewrite_input_seeds(expected,parameters['seeds'])
        if digest(expected)!=digest(workspace/'Ang Repouso/in.repouso'):
            raise ValueError('trial input does not match reviewed protocol and factors')
    pins={}
    for rel,sha in REVIEWED_ASSETS.items():
        if rel.endswith('/in.repouso'): continue
        if digest(workspace/rel)!=sha: raise ValueError('untrusted trial asset: '+rel)
        pins[rel]=sha
    pins['Ang Repouso/in.repouso']=digest(workspace/'Ang Repouso/in.repouso')
    pins['parameters.json']=digest(workspace/'parameters.json')
    return pins,point


def implementation_pins():
    root=Path(__file__).parent
    return {n:digest(root/n) for n in ['protocol_eligibility.py','rocky_io.py','response_policy.py','calibration_controller.py']}


HISTORICAL_IMPLEMENTATION_PINS = {
    'protocol_eligibility.py': 'b8b3c1ec576f376874983d6bc971758013db0e56416e62b5bf78e4bd0e77d87d',
    'rocky_io.py': '38c744668e1ad7e5100cccc35ce54c5d422b4c891a4bb582b25e8e6c183c7870',
    'response_policy.py': 'c345967d6a2ad21207d561f106a3d534b0f882eaa8d875dfde702b479a2663e5',
    'calibration_controller.py': 'de8b2caaf8d047853c7b074d46e09b7c7fdda69b8a1285a38a031114721e1188',
}

def trusted_implementation(pins):
    return pins == implementation_pins() or pins == HISTORICAL_IMPLEMENTATION_PINS


def completed_endpoint(workspace, dt=5e-5):
    import re
    if dt not in (5e-5, 1e-4):
        return False
    steps = 200000 if dt == 5e-5 else 100000
    text=(Path(workspace)/'repose.log').read_text(errors='replace')
    return (text.count('Loop time of ') >= 3 and 'ERROR' not in text
            and re.search(r'^\s*'+str(steps)+r'\s+10(?:\.0+)?\s+\d+\s+',text,re.M) is not None
            and 'System init for write_restart' in text)


def record_fresh_repose(controller, workspace, raw):
    """Called only by execute_trial after successful solver and final measurement."""
    from .rocky_io import json_ready
    workspace=Path(workspace).resolve()
    pins,point=_input_pins(controller,workspace)
    snapshot=workspace/'Ang Repouso/post/repouso_10s.vtk'
    if Path(raw['source_path']).resolve()!=snapshot or raw['source_sha256']!=digest(snapshot):
        raise ValueError('only the final protocol snapshot may be admitted')
    if raw.get('config_used')!=controller.config['measurement']:
        raise ValueError('measurement configuration provenance mismatch')
    log=workspace/'repose.log'
    dt=overlay_config(controller.project_root)['timestep_s']
    if not log.is_file() or not completed_endpoint(workspace,dt):
        raise ValueError('successful completed solver endpoint required')
    trial_id=int(workspace.name.removeprefix('trial_'))
    receipt={'schema_version':1,'trial_id':trial_id,'implementation_sha256':implementation_pins(),'protocol_revision':overlay_config(controller.project_root)['protocol_revision'],
             'pins':pins,'parameters':point,'snapshot_sha256':digest(snapshot),
             'solver_log_sha256':digest(log),'measurement':json_ready(raw)}
    (workspace/'repose_admission.json').write_text(json.dumps(receipt,allow_nan=False,indent=2))
    return receipt


def _admitted(controller,row):
    w=Path(os.path.abspath(row['workspace']))
    # CSVs copied from another installation can contain absolute paths into a
    # live campaign. Never open even a receipt until the workspace is confined.
    if not w.is_relative_to(Path(controller.run_root).resolve()):
        return False
    w = w.resolve()
    if not w.is_relative_to(Path(controller.run_root).resolve()):
        return False
    receipt=json.loads((w/'repose_admission.json').read_text())
    if receipt.get('schema_version') != 1 or not trusted_implementation(receipt['implementation_sha256']): return False
    historical = receipt['implementation_sha256'] == HISTORICAL_IMPLEMENTATION_PINS
    if controller.config.get('fresh_finite_independent') is True and historical:
        return False
    pins,point=_input_pins(controller,w,historical=historical)
    if receipt['pins']!=pins or receipt['parameters']!=point: return False
    dt=5e-5 if historical else overlay_config(controller.project_root)['timestep_s']
    if not completed_endpoint(w,dt): return False
    trial_id=int(w.name.removeprefix('trial_'))
    if int(row['trial_id'])!=trial_id or receipt['trial_id']!=trial_id: return False
    if any(not math.isclose(float(row[k]),float(v),rel_tol=0,abs_tol=1e-10) for k,v in point.items()): return False
    raw=receipt['measurement']; snapshot=w/'Ang Repouso/post/repouso_10s.vtk'
    if receipt['protocol_revision']!=overlay_config(controller.project_root)['protocol_revision']: return False
    if (receipt['snapshot_sha256']!=digest(snapshot) or raw['source_sha256']!=digest(snapshot)
        or Path(raw['source_path']).resolve()!=snapshot or receipt['solver_log_sha256']!=digest(w/'repose.log')
        or raw['config_used']!=controller.config['measurement']): return False
    recomputed=controller._measure_snapshot(snapshot,'repose')
    adopted=recomputed['optimization_response']['adopted_angle_deg']
    if adopted != raw['optimization_response']['adopted_angle_deg']: return False
    actual=row.get('repose_deg')
    if adopted is None: return actual in (None,'')
    return math.isclose(float(actual),float(adopted),rel_tol=0,abs_tol=1e-10)


def interpret_for_controller(controller, row, test, target=None):
    outcome=interpret_response(row,test,target)
    if test!='repose': return outcome
    root=getattr(controller,'project_root',None)
    if root is None: return outcome
    cfg=overlay_config(root)
    if not cfg or int(cfg.get('schema_version',1))<2: return outcome
    try:
        admitted=_admitted(controller,row)
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        admitted=False
    if admitted:
        return dict(outcome,protocol_eligible=True,protocol_revision=cfg['protocol_revision'],
                    basis='verified_trial_inputs_and_final_measurement')
    return dict(outcome,outcome_code='protocol_quarantined',label_pt='Repouso em quarentena de protocolo',
                basis='schema2_requires_verified_measurement_provenance',protocol_eligible=False,
                protocol_revision=cfg.get('protocol_revision'),angle_deg=None,error_deg=None,
                regression_included=False,physical_classifier_label=None,causal_cohesion_verified=False)
