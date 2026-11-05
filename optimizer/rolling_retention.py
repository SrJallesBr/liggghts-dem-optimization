"""Opt-in V4 trial retention, after consumers finish. Never a campaign sweeper.

Prepared manifests are durable write-ahead journals. A retry checks all remaining
files against that journal before resuming. Deleted velocities cannot be replayed;
offline re-evaluation checks the immutable per-sample audit and native thermo.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
from optimizer.rolling_protocol import fraction_denominator, population_fraction, threshold_crossed, PRESENT_VERSION

OWNER = 'rolling_retention_owner.json'
MANIFEST = 'rolling_retention.json'
STEPS = tuple(range(100000, 320001, 1000))


def safe(path, *, regular=False):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue  # New output paths still require checking existing parents.
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('symlink forbidden: '+str(part))
        if getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('reparse point/junction forbidden: '+str(part))
    if regular:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('expected private regular file: '+str(path))
    return path


def descriptor(path):
    path = safe(path, regular=True)
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(filename=path.name, bytes=path.stat().st_size, sha256=sha)


def check(post, item):
    name = item['filename']
    if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('external or malformed evidence filename')
    if descriptor(post/name) != item:
        raise ValueError('evidence size/hash mismatch: '+name)


def sync_directory(path):
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def durable(path, text):
    path = safe(path)
    if path.exists(): safe(path, regular=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8', newline='') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def save(path, value):
    durable(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


def initialize_retention(trial_root, protocol_version="V5"):
    """Only called on a fresh prepared trial, before any solver output exists."""
    trial = safe(trial_root)
    if not re.fullmatch(r'trial_\d+', trial.name) or {'tests','fixtures','.git'} & set(trial.parts):
        raise ValueError('retention requires a fresh named trial')
    post = safe(trial/'Ang Rolamento/post')
    if post.exists() or (trial/OWNER).exists():
        raise ValueError('cannot authorize retention of existing output')
    value = dict(schema=1, protocol_version=protocol_version, speed_threshold_m_s=.001,
                 fraction_denominator=fraction_denominator(protocol_version),
                 trial_root=str(trial), post=str(post), token=uuid.uuid4().hex)
    with (trial/OWNER).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    sync_directory(trial)


def ownership(post, trial_root):
    post, trial = safe(post), safe(trial_root)
    if post != trial/'Ang Rolamento/post':
        raise ValueError('post outside owned rolling trial root')
    if post.resolve() != trial.resolve()/'Ang Rolamento/post':
        raise ValueError('post outside canonical physical rolling trial scope')
    safe(trial/OWNER, regular=True)
    owner = json.loads((trial/OWNER).read_text())
    if not isinstance(owner, dict): raise ValueError('retention owner must be a JSON object')
    if owner.get('protocol_version') not in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'): raise ValueError('unsupported owner version')
    if (owner.get('schema'), owner.get('speed_threshold_m_s'),
        owner.get('trial_root'), owner.get('post')) != (1, .001, str(trial), str(post)):
        raise ValueError('missing or incompatible retention ownership')
    return post, owner


def inventory(post, allowed):
    for path in post.iterdir():
        # A symlink of any kind in post blocks the whole transaction.
        safe(path)
        if path.name.endswith('.dump') or path.name.startswith(('particles_', 'rolling_event_particles')):
            if path.name not in allowed:
                raise ValueError('unexpected particle filename: '+path.name)
            safe(path, regular=True)


def read_retained_samples(post):
    """Read-only recovery/re-evaluation with no fallback around corrupt evidence."""
    from optimizer.rolling_measurement import read_dump, total_speed_moving, ids_sha256
    post = safe(post)
    safe(post/MANIFEST, regular=True)
    value = json.loads((post/MANIFEST).read_text())
    if not isinstance(value, dict): raise ValueError('retention manifest must be a JSON object')
    version=value.get('protocol_version')
    if version not in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') or (value.get('schema'), value.get('speed_threshold_m_s')) != (1, .001):
        raise ValueError('unsupported retention protocol')
    denominator = fraction_denominator(version)
    if version == PRESENT_VERSION and value.get('fraction_denominator') != denominator:
        raise ValueError('retention denominator/version mismatch')
    if value.get('state') not in ('prepared', 'complete'):
        raise ValueError('incomplete retention evidence')
    selected = value['selected_dump']
    step = value['measured_step']
    dt = value.get('dt_s', .0001)
    if type(dt) not in (int, float) or dt not in (.0001, .00005) or (dt == .00005 and version != PRESENT_VERSION):
        raise ValueError('unsupported retention timestep')
    scale = 2 if dt == .00005 else 1
    cadence = 1000 * scale
    if type(step) is not int or not (100000*scale <= step <= 400000*scale) or step % cadence:
        raise ValueError('invalid retained event timestep')
    steps = tuple(range(10000*scale, step+1, cadence)) if version in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') else STEPS
    tilt_start = 100000*scale if version in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') else 170000
    if type(step) is not int or step not in steps or selected['filename'] != f'particles_{step}.dump':
        raise ValueError('retention selected dump clock mismatch')
    if (value['measured_time_s'] != step*dt or
            not math.isclose(value['measured_angle_deg'], (step-tilt_start)*dt*3, rel_tol=0., abs_tol=1e-7)):
        raise ValueError('retention measured angle/time mismatch')
    names = {'rolling_thermo.log','rolling_event.txt',f'rolling_event_base_{step}.stl',
             'rolling_measurement.svg','rolling_measurement.png','rolling_samples.csv','rolling_validation.json'}
    if not isinstance(value['evidence'], dict): raise ValueError('retained evidence must be a JSON object')
    if set(value['evidence']) != names or any(item['filename'] != name for name,item in value['evidence'].items()):
        raise ValueError('missing or unexpected retained evidence binding')
    expected = {f'particles_{s}.dump' for s in steps} | {'rolling_event_particles.dump'}
    deleted = value['deleted_files']
    if len(deleted) != len(steps) or {d['filename'] for d in deleted} != expected-{selected['filename']}:
        raise ValueError('invalid deletion journal')
    inventory(post, expected if value['state'] == 'prepared' else {selected['filename']})
    for item in [selected, *value['evidence'].values()]: check(post, item)
    if value['state'] == 'prepared':
        for item in deleted:
            if (post/item['filename']).exists(): check(post, item)
    baseline = value['baseline_ids']
    if not baseline or len(set(baseline)) != len(baseline) or any(type(i) is not int or i <= 0 for i in baseline):
        raise ValueError('invalid retained baseline IDs')
    rows = {}
    survivors = set(baseline)
    selected_ids = None
    with (post/'rolling_samples.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            for key in ('step','N0','alive','moving','lost','bytes'): row[key] = int(row[key])
            for key in ('time_s','angle_deg','fraction'): row[key] = float(row[key])
            if (version == PRESENT_VERSION and row.get('fraction_denominator') != denominator) or row.get('fraction_denominator', denominator) != denominator:
                raise ValueError('retained audit denominator/version mismatch')
            clock = row['step']
            if clock in rows or clock not in steps: raise ValueError('duplicate/invalid audit clock')
            losses = json.loads(row['new_lost_ids'])
            if (len(set(losses)) != len(losses) or not set(losses) <= survivors
                    or any(type(i) is not int for i in losses) or (clock == steps[0] and losses)):
                raise ValueError('invalid retained particle losses')
            survivors -= set(losses)
            if (row['N0'], row['alive'], row['lost']) != (len(baseline), len(survivors), len(baseline)-len(survivors)):
                raise ValueError('retained audit population mismatch')
            if not 0 <= row['moving'] <= row['alive'] or row['fraction'] != population_fraction(row['moving'], row['alive'], len(baseline), version):
                raise ValueError('invalid retained fraction')
            if (row['time_s'] != clock*dt or not math.isclose(row['angle_deg'], max(0,clock-tilt_start)*dt*3, abs_tol=1e-7)
                    or row['ids_sha256'] != ids_sha256(survivors) or row['filename'] != f'particles_{clock}.dump'):
                raise ValueError('retained audit geometry/ID clock mismatch')
            if clock == step: selected_ids = survivors.copy()
            rows[clock] = row
    if tuple(rows) != steps: raise ValueError('missing or unordered retained sample audit')
    first = next((r['step'] for r in rows.values() if r['step'] >= tilt_start and threshold_crossed(r['moving'], r['alive'], r['N0'], version)), None)
    if first != step: raise ValueError('retained event is not first crossing')
    clock, particles, sha = read_dump(post/selected['filename'], geometry=True)
    row = rows[step]
    if (clock != step or set(particles) != selected_ids or sha != row['sha256'] or selected['bytes'] != row['bytes']
            or sum(total_speed_moving(p) for p in particles.values()) != row['moving']):
        raise ValueError('retained event dump disagrees with audit')
    for item in deleted:
        if item['filename'].startswith('particles_'):
            r = rows[int(item['filename'][10:-5])]
            if item != {k:r[k] for k in ('filename','bytes','sha256')}:
                raise ValueError('deletion journal/audit mismatch')
    return dict(rows=rows, baseline_ids=baseline, measured_step=step, manifest=value)


def finalize_rolling_trial(post, *, trial_root):
    """Explicit last-consumer action. Any failed validation retains diagnostic dumps."""
    from optimizer.rolling_measurement import measure_rolling
    from optimizer.rolling_visuals import read_rolling_event, render_rolling_snapshot
    try:
        post, owner = ownership(post, trial_root)
        if (post/MANIFEST).exists():
            archive = read_retained_samples(post)
            value = archive['manifest']
            if value['owner_token'] != owner['token']: raise ValueError('retention owner mismatch')
            metric = measure_rolling(post)
            if metric['status'] != 'measured': raise ValueError('retained measurement invalid: '+str(metric.get('reason')))
            read_rolling_event(post, metric)
            if value['state'] == 'complete': return dict(status='already_retained', manifest=str(post/MANIFEST))
        else:
            metric = measure_rolling(post)
            scale = 2 if metric.get('dt_s') == .00005 and owner['protocol_version'] == PRESENT_VERSION else 1
            steps = tuple(range(10000*scale, 400000*scale+1, 1000*scale)) if owner['protocol_version'] in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') else STEPS
            expected = {f'particles_{s}.dump' for s in steps} | {'rolling_event_particles.dump'}
            inventory(post, expected)
            # Check every output target before the renderer can touch it.
            for name in ('rolling_measurement.svg','rolling_measurement.png','rolling_validation.json','rolling_samples.csv',MANIFEST):
                if (post/name).exists(): safe(post/name, regular=True)
            if metric.get('protocol_version') != owner['protocol_version'] or metric['status'] != 'measured':
                raise ValueError('retention requires measured full-run V4 data: '+str(metric.get('reason')))
            event = read_rolling_event(post, metric)
            snapshot = render_rolling_snapshot(post, metric)
            if snapshot['status'] != 'rendered': raise ValueError('image render failed: '+str(snapshot.get('reason')))
            # Decode pixels, not just a renderer success flag or file extension.
            from PIL import Image
            with Image.open(post/'rolling_measurement.png') as image:
                image.load()
                if image.format != 'PNG' or image.size != (1600,1060): raise ValueError('invalid rolling PNG')
            svg = (post/'rolling_measurement.svg').read_text()
            if 'protractor-tick' not in svg or 'data-kind="particle"' not in svg or 'base-triangle' not in svg:
                raise ValueError('image missing protractor or physical scene')
            selected = post/f'particles_{event["step"]}.dump'
            rows = metric['sample_audit']
            expected = {row['filename'] for row in rows} | {'rolling_event_particles.dump'}
            inventory(post, expected)
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            durable(post/'rolling_samples.csv', stream.getvalue())
            save(post/'rolling_validation.json', dict(metric, snapshot=snapshot))
            evidence_names = ['rolling_thermo.log','rolling_event.txt',Path(event['paths']['base']).name,
                              'rolling_measurement.svg','rolling_measurement.png','rolling_samples.csv','rolling_validation.json']
            value = dict(schema=1, state='prepared', protocol_version=owner['protocol_version'], speed_threshold_m_s=.001,
                         fraction_denominator=fraction_denominator(owner['protocol_version']),
                         owner_token=owner['token'], dt_s=metric['dt_s'], measured_step=event['step'], measured_time_s=event['step']*metric['dt_s'],
                         measured_angle_deg=metric['response_angle_deg'], baseline_ids=metric['baseline_ids'],
                         selected_dump=descriptor(selected), evidence={name:descriptor(post/name) for name in evidence_names},
                         deleted_files=[descriptor(post/name) for name in sorted(expected-{selected.name})])
            # Persist all inputs, pixels, report, CSV, then journal, before unlink.
            for name in [selected.name, *evidence_names, *[d['filename'] for d in value['deleted_files']]]:
                with safe(post/name, regular=True).open('rb') as handle: os.fsync(handle.fileno())
            save(post/MANIFEST, value)
            read_retained_samples(post)
        # Verify the entire remaining deletion set before removing its first file.
        for item in value['deleted_files']:
            if (post/item['filename']).exists(): check(post, item)
        for item in value['deleted_files']:
            path = post/item['filename']
            if path.exists():
                check(post, item)
                path.unlink()
        sync_directory(post)
        value['state'] = 'complete'
        save(post/MANIFEST, value)
        read_retained_samples(post)
        return dict(status='retained', manifest=str(post/MANIFEST), selected_dump=value['selected_dump'], deleted_count=len(value['deleted_files']))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return dict(status='retention_skipped', reason=f'{type(exc).__name__}: {exc}')
