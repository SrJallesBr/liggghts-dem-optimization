"""Narrow, one-use repose-only validation. Does not authorize a DOE."""
from pathlib import Path
import hashlib


def valid_single_validation(project, config, disk_config, approval):
    pins=approval.get('deployed_sha256',{})
    return (
        approval.get('scope')=='repose-single-validation'
        and Path(approval.get('project_root','')).resolve()==project
        and approval.get('requested_attempts')==1
        and approval.get('doe_authorized') is False
        and bool(approval.get('authorization_statement'))
        and config==disk_config
        and config.get('tests')==['repose']
        and config.get('max_trials')==1
        and approval.get('input_relative')=='Ang Repouso/in.repouso'
        and not any((project/name).exists() for name in ['results.csv','results','runs','runtime/future_repose_overlay.json'])
        and isinstance(pins,dict)
        and set(pins) == {'Ang Repouso/in.repouso', 'Ang Repouso/initial_particles.data',
                          'Ang Repouso/meshes/Container_Cylinder.stl', 'Ang Repouso/meshes/Upper_Tray.stl'}
        and all(not Path(k).is_absolute() and '..' not in Path(k).parts
                and (project/k).is_file()
                and hashlib.sha256((project/k).read_bytes()).hexdigest()==v for k,v in pins.items())
    )


def check_single_invocation(project, workspace, directory, input_name, command):
    import json
    project=Path(project).resolve()
    approval=json.loads((project/'launch_authorization.json').read_text())
    if (Path(workspace).resolve()!=project or directory!='Ang Repouso' or input_name!='in.repouso'
            or command!=approval.get('solver_command',[])+['-in','in.repouso']):
        raise RuntimeError('HOLD: single validation cannot launch another workspace/deck/solver')


def consume_single_validation(project):
    path=project/'runtime/repose_single_validation_consumed.json'
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open('x',encoding='utf-8') as f:f.write('{"attempts_consumed": 1, "doe_authorized": false}\n')
    except FileExistsError as exc:
        raise RuntimeError('HOLD: repose single validation already consumed') from exc
