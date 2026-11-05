"""Streaming collective translational motion, not microscopic pure rolling.

Memory is O(initial population), independent of dump history length. Every
sample through the protocol endpoint is checked, even after a crossing.
"""
from pathlib import Path
import hashlib
import math
import re
import json
from optimizer.rolling_protocol import fraction_denominator, population_fraction, threshold_crossed


def ids_sha256(ids):
    return hashlib.sha256(','.join(map(str, sorted(ids))).encode('ascii')).hexdigest()

SPEED_THRESHOLD = .001
LEGACY_SPEED_THRESHOLD = .00156604597634
START_STEP, END_STEP, SAMPLE_STEPS = 100000, 250000, 1000
DT = 1e-4


def read_dump(path, *, geometry=False, allow_empty=False):
    """Read exactly one ASCII custom dump, rejecting ambiguous/truncated data."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        def line():
            data = handle.readline()
            if not data:
                raise ValueError('truncated dump')
            digest.update(data)
            return data.decode('ascii').strip()

        if line() != 'ITEM: TIMESTEP': raise ValueError('malformed dump clock header')
        step_text = line()
        if not re.fullmatch(r'\d+', step_text): raise ValueError('malformed dump clock')
        step = int(step_text)
        if line() != 'ITEM: NUMBER OF ATOMS': raise ValueError('malformed population header')
        count_text = line()
        if not (re.fullmatch(r'[1-9]\d*', count_text) or (allow_empty and count_text == '0')):
            raise ValueError('empty or malformed population')
        count = int(count_text)
        if line() != 'ITEM: BOX BOUNDS ff ff ff': raise ValueError('unsupported box bounds')
        for _ in range(3):
            bounds = list(map(float, line().split()))
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
                raise ValueError('invalid box bounds')
        header = line().split()
        fields = header[2:]
        if header[:2] != ['ITEM:', 'ATOMS'] or not {'id', 'vz'} <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError('missing or duplicate atom fields')
        if geometry and not set('id type x y z radius vx vy vz omegax omegay omegaz'.split()) <= set(fields):
            raise ValueError('event dump missing geometry or angular velocity fields')
        particles = {}
        for _ in range(count):
            values = line().split()
            if len(values) != len(fields): raise ValueError('truncated or malformed atom fields')
            numeric = list(map(float, values))
            if not all(map(math.isfinite, numeric)): raise ValueError('nonfinite atom data')
            atom_text = values[fields.index('id')]
            if not re.fullmatch(r'[1-9]\d*', atom_text): raise ValueError('invalid particle ID')
            atom = int(atom_text)
            if atom in particles: raise ValueError('duplicate particle ID')
            row = dict(zip(fields,numeric))
            if geometry and (row['radius'] <= 0 or row['type'] <= 0 or not row['type'].is_integer()):
                raise ValueError('invalid particle radius or type')
            particles[atom] = row if geometry else row['vz']
        remainder = handle.read(1)
        if remainder: raise ValueError('extra data or duplicate dump clock')
    return step, particles, digest.hexdigest()


def _empty_result():
    return dict(status='invalid_data', response_angle_deg=None,
                  observed_angle_deg=None, observed_fraction=None,
                  window_start_step=None, window_end_step=None, samples=0,
                  initial_particle_count=None, final_particle_count=None,
                  first_crossing=None,
                  loss_before_window=False, lost_particle_count=0,
                  speed_threshold_m_s=LEGACY_SPEED_THRESHOLD, fraction_threshold=.95,
                  consecutive_samples=5, sample_interval_s=.1, dt_s=DT,
                  tilt_start_s=10., tilt_rate_deg_s=3., maximum_angle_deg=45.,
                  microscopic_pure_rolling_verified=False,
                  limitation='Collective downhill motion; no proof of microscopic pure rolling.',
                  full_run_loss_policy='reject even if a window was observed before loss')


def _measure_rolling_dumps(post):
    result = _empty_result()
    result.update(measurement_source='legacy_particle_dumps', particle_identity_verified=True)
    fingerprint = hashlib.sha256()
    baseline = None
    try:
        paths = list(Path(post).glob('particles_*.dump'))
        if not paths: raise ValueError('missing initial time-10 baseline dumps')
        def clock(path):
            match = re.fullmatch(r'particles_(\d+)\.dump', path.name)
            if not match: raise ValueError('malformed filename clock')
            return int(match[1])
        paths.sort(key=clock)
        streak = 0
        expected = START_STEP
        for path in paths:
            step, particles, sha = read_dump(path)
            if step != clock(path): raise ValueError('filename/dump clock mismatch')
            if baseline is None and step != START_STEP: raise ValueError('missing initial time-10 baseline')
            if step != expected or step > END_STEP: raise ValueError('clock gap, duplicate or unordered sample')
            if baseline is None:
                baseline = set(particles)
                result['initial_particle_count'] = len(baseline)
            result['final_particle_count'] = len(particles)
            if particles.keys() != baseline:
                result['lost_particle_count'] = len(baseline - particles.keys())
                result['loss_before_window'] = result['window_start_step'] is None
                raise ValueError('particle population changed; denominator must not shrink')
            moving = sum(v < -LEGACY_SPEED_THRESHOLD for v in particles.values())
            fraction = moving / len(baseline)
            if 20*moving >= 19*len(baseline) and result['first_crossing'] is None:
                result['first_crossing'] = dict(step=step, angle_deg=(step-START_STEP)*DT*3,
                    moving=moving, N0=len(baseline),
                    alive=len(particles), lost=0)
            streak = streak + 1 if fraction >= .95 else 0
            if streak == 1:
                start_fraction = fraction
            if streak == 5 and result['window_start_step'] is None:
                first = step - 4*SAMPLE_STEPS
                result.update(observed_angle_deg=(first-START_STEP)*DT*3,
                              observed_fraction=start_fraction,
                              window_start_step=first, window_end_step=step)
            fingerprint.update(f'{step}:{sha}\n'.encode('ascii'))
            result['samples'] += 1
            expected += SAMPLE_STEPS
        if expected != END_STEP + SAMPLE_STEPS: raise ValueError('missing full-run endpoint at 25 s')
        result.update(status='measured' if result['window_start_step'] is not None else 'censored',
                      response_angle_deg=result['observed_angle_deg'],
                      reason=None if result['window_start_step'] is not None else 'no sustained threshold through 45 degrees')
    except (OSError, ValueError, UnicodeError, OverflowError, KeyError, TypeError) as exc:
        result.update(status='invalid_data', response_angle_deg=None, reason=str(exc))
    result['source_sha256'] = fingerprint.hexdigest()
    return result


THERMO_HEADER = ('Step Time Atoms v_angulo_gravidade_graus '
                 'c_moving_count v_initial_count').split()


def total_speed_moving(particle, threshold=SPEED_THRESHOLD):
    """One strict 0/1 vote per grain; no average or sustained-motion rule."""
    return math.sqrt(particle['vx']**2 + particle['vy']**2 + particle['vz']**2) > threshold


def _measure_rolling_thermo(path):
    """Count threshold crossings, never mean/sum speeds.

    V3 independently verifies every particle dump; legacy counts alone prove
    population size, not particle identity. Boundary losses are
    permitted; absent grains never count as moving and N0 never shrinks.
    Insertion/deletion commands during tilt are forbidden by preflight.
    Only identical shared endpoints between separate run blocks are deduplicated.
    """
    result = _empty_result()
    result.update(measurement_source='aggregate_thermo', particle_identity_verified=False,
                  native_event_protocol=False,
                  population_integrity='fixed initial N0; boundary losses allowed; gains invalid',
                  full_run_loss_policy='boundary losses allowed; missing particles are not moving')
    source_sha256 = None
    dt, sample_steps = DT, SAMPLE_STEPS
    start_step = START_STEP
    expected, streak = start_step, 0
    tilt_start, end_step = start_step, END_STEP
    active, first_in_block, completed, boundary = False, False, False, False
    previous = None
    previous_ids = None
    dump_fingerprint = hashlib.sha256()
    sample_audit = []
    retained = None
    try:
        if (Path(path).parent/'rolling_retention.json').exists():
            from optimizer.rolling_retention import read_retained_samples
            retained = read_retained_samples(Path(path).parent)
        with Path(path).open('rb') as handle:
            source_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
            handle.seek(0)
            for raw in handle:
                text = raw.decode('ascii').strip()
                if text.startswith('ERROR'):
                    raise ValueError('solver error in thermo log: ' + text)
                if text.startswith('ROLLING_EVENT_PROTOCOL_') and text not in (
                        'ROLLING_EVENT_PROTOCOL_V1', 'ROLLING_EVENT_PROTOCOL_V2', 'ROLLING_EVENT_PROTOCOL_V3', 'ROLLING_EVENT_PROTOCOL_V4', 'ROLLING_EVENT_PROTOCOL_V5', 'ROLLING_EVENT_PROTOCOL_V6_LIFT_HEIGHT_3S', 'ROLLING_EVENT_PROTOCOL_V7_PRESENT_POPULATION'):
                    raise ValueError('unknown rolling protocol marker')
                if text in ('ROLLING_EVENT_PROTOCOL_V1', 'ROLLING_EVENT_PROTOCOL_V2', 'ROLLING_EVENT_PROTOCOL_V3', 'ROLLING_EVENT_PROTOCOL_V4', 'ROLLING_EVENT_PROTOCOL_V5', 'ROLLING_EVENT_PROTOCOL_V6_LIFT_HEIGHT_3S', 'ROLLING_EVENT_PROTOCOL_V7_PRESENT_POPULATION'):
                    if previous is not None or result['native_event_protocol']:
                        raise ValueError('late or duplicate rolling protocol marker')
                    result['protocol_version'] = text[len('ROLLING_EVENT_PROTOCOL_'):]
                    result['fraction_denominator'] = fraction_denominator(result['protocol_version'])
                    if result['protocol_version'] in ('V2', 'V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                        tilt_start, end_step = 170000, 320000
                        result['tilt_start_s'] = tilt_start*dt
                    if result['protocol_version'] in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                        start_step = expected = 10000
                        tilt_start, end_step = 100000, 400000
                        result.update(tilt_start_s=10., maximum_angle_deg=90.)
                    if result['protocol_version'] in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                        result.update(consecutive_samples=1, velocity_metric='total_speed',
                                      event_rule='first_sampled_crossing',
                                      measurement_source='verified_total_speed_particle_dumps',
                                      limitation='Collective translational motion; no proof of microscopic pure rolling.')
                        if result['protocol_version'] in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                            result['speed_threshold_m_s'] = SPEED_THRESHOLD
                    result['native_event_protocol'] = True
                    continue
                if text == 'ROLLING_THERMO_COMPLETE':
                    if completed:
                        raise ValueError('duplicate thermo completion marker')
                    completed = True
                    active = False
                    continue
                if completed and (text.split() in (THERMO_HEADER, ['Step', 'Time', 'Atoms', 'angulo_g', 'moving_c', 'initial_'])
                                  or re.match(r'^[+\-\d.]', text)
                                  or re.fullmatch(r'\S+(?:\s+[+\-\d.]\S*){5}', text)):
                    raise ValueError('thermo data after completion marker')
                # LIGGGHTS-PUBLIC 3.8 drops c_/v_ and truncates custom labels.
                if text.split() in (THERMO_HEADER, ['Step', 'Time', 'Atoms', 'angulo_g', 'moving_c', 'initial_']):
                    if previous is not None and not boundary:
                        raise ValueError('thermo header without a completed run boundary')
                    boundary = False
                    active, first_in_block = True, True
                    continue
                if text.startswith('Loop time'):
                    boundary = active
                    active = False
                    continue
                if not active or not text:
                    continue
                fields = text.split()
                if len(fields) != len(THERMO_HEADER):
                    raise ValueError('malformed thermo fields')
                values = tuple(map(float, fields))
                if not all(map(math.isfinite, values)):
                    raise ValueError('nonfinite thermo data')
                step_f, time, count_f, angle, moving_f, initial_f = values
                if any(not v.is_integer() or abs(v) > 2**53 for v in (step_f, count_f, moving_f, initial_f)):
                    raise ValueError('noninteger thermo clock or population counts')
                step, count, moving, initial = map(int, (step_f, count_f, moving_f, initial_f))
                if initial <= 0 or count < 0 or not 0 <= moving <= count:
                    raise ValueError('invalid population or moving count')
                if 'thermo_time_origin_s' not in result:
                    # V7 has two supported, physically identical clocks.  A missing
                    # baseline is never inferred from arbitrary later samples.
                    if result.get('protocol_version') == 'V7_PRESENT_POPULATION' and step == 20000:
                        dt, sample_steps = 5e-5, 2000
                        start_step = expected = 20000
                        tilt_start, end_step = 200000, 800000
                    result.update(dt_s=dt, sample_interval_s=sample_steps*dt)
                    if step != start_step:
                        raise ValueError('missing initial protocol baseline')
                    if time not in (0., start_step*dt):
                        raise ValueError('unsupported thermo time origin/clock')
                    result['thermo_time_origin_s'] = time
                expected_time = result['thermo_time_origin_s'] + (step-start_step)*dt
                if not math.isclose(time, expected_time, abs_tol=1e-7, rel_tol=0.):
                    raise ValueError('thermo time/clock mismatch')
                if not math.isclose(angle, max(0, step-tilt_start)*dt*3, abs_tol=1e-7, rel_tol=0.):
                    raise ValueError('thermo angle/clock mismatch')
                if previous is not None and step == previous[0]:
                    if first_in_block and values == previous:
                        first_in_block = False
                        continue
                    raise ValueError('duplicate or conflicting thermo clock')
                first_in_block = False
                if result['initial_particle_count'] is None:
                    if step != start_step:
                        raise ValueError('missing initial protocol baseline')
                    result['initial_particle_count'] = initial
                if step != expected or step > end_step:
                    raise ValueError('clock gap, duplicate or unordered sample')
                baseline = result['initial_particle_count']
                result['final_particle_count'] = count
                if initial != baseline or count > (previous[2] if previous else baseline):
                    raise ValueError('particle population gain or changed initial denominator')
                if result.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                    if retained is not None:
                        if result['protocol_version'] not in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                            raise ValueError('retention requires V4/V5')
                        row = retained['rows'][step]
                        if (row['N0'], row['alive'], row['moving']) != (baseline, count, moving):
                            raise ValueError('retained audit/thermo count mismatch')
                        sample_audit.append(row)
                        dump_fingerprint.update(f'{step}:{row["sha256"]}\n'.encode('ascii'))
                    else:
                        dump_step, particles, sha = read_dump(Path(path).parent/f'particles_{step}.dump',
                                                              geometry=True, allow_empty=True)
                        if dump_step != step or len(particles) != count:
                            raise ValueError('particle dump/thermo clock or population mismatch')
                        ids = set(particles)
                        if previous_ids is None:
                            if len(ids) != baseline:
                                raise ValueError('initial particle dump must establish fixed N0')
                        elif not ids <= previous_ids:
                            raise ValueError('particle ID gain or reappearance after boundary loss')
                        if result['protocol_version'] in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                            if previous_ids is None:
                                result['baseline_ids'] = sorted(ids)
                            sample_audit.append(dict(step=step, time_s=step*dt, angle_deg=angle,
                                N0=baseline, alive=count, moving=moving, lost=baseline-count,
                                fraction=population_fraction(moving, count, baseline, result['protocol_version']),
                                fraction_denominator=result['fraction_denominator'], filename=f'particles_{step}.dump',
                                bytes=(Path(path).parent/f'particles_{step}.dump').stat().st_size,
                                sha256=sha, ids_sha256=ids_sha256(ids),
                                new_lost_ids=json.dumps(sorted((previous_ids if previous_ids is not None else ids)-ids), separators=(',', ':'))))
                        previous_ids = ids
                        if sum(total_speed_moving(p, result['speed_threshold_m_s']) for p in particles.values()) != moving:
                            raise ValueError('particle total speed count/thermo mismatch')
                        dump_fingerprint.update(f'{step}:{sha}\n'.encode('ascii'))
                result['lost_particle_count'] = baseline-count
                if count < baseline and result['window_start_step'] is None:
                    result['loss_before_window'] = True
                fraction = population_fraction(moving, count, baseline, result.get('protocol_version'))
                if step >= tilt_start and threshold_crossed(moving, count, baseline, result.get('protocol_version')) and result['first_crossing'] is None:
                    result['first_crossing'] = dict(step=step, angle_deg=max(0, step-tilt_start)*dt*3,
                        moving=moving, N0=baseline, alive=count, lost=baseline-count)
                streak = streak + 1 if step >= tilt_start and threshold_crossed(moving, count, baseline, result.get('protocol_version')) else 0
                if streak == 1:
                    start_fraction = fraction
                if streak == result['consecutive_samples'] and result['window_start_step'] is None:
                    first = step - (result['consecutive_samples']-1)*sample_steps
                    result.update(observed_angle_deg=(first-tilt_start)*dt*3,
                                  observed_fraction=start_fraction,
                                  window_start_step=first, window_end_step=step)
                previous = values
                expected += sample_steps
                result['samples'] += 1
        if result.get('protocol_version') in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') and result['first_crossing'] is not None:
            end_step = result['first_crossing']['step']
        if expected != end_step + sample_steps:
            raise ValueError(f'missing full-run endpoint at {end_step*dt:g} s')
        if not completed:
            raise ValueError('missing thermo completion marker')
        if result.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') and retained is None:
            expected_names = {f'particles_{s}.dump' for s in range(start_step, end_step+1, sample_steps)}
            if {p.name for p in Path(path).parent.glob('particles_*.dump')} != expected_names:
                raise ValueError('particle dump inventory has extra, missing or malformed filename clocks')
        result.update(status='measured' if result['window_start_step'] is not None else 'censored',
                      response_angle_deg=result['observed_angle_deg'],
                      reason=None if result['window_start_step'] is not None else 'no sustained threshold through 45 degrees')
        if result.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
            result.update(particle_identity_verified=True, particle_dumps_sha256=dump_fingerprint.hexdigest())
            if result['status'] == 'censored':
                result['reason'] = f'no sampled total-speed crossing through {result["maximum_angle_deg"]:g} degrees'
        if result.get('protocol_version') in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
            result['sample_audit'] = sample_audit
            if retained is not None:
                result['baseline_ids'] = retained['baseline_ids']
                result['verification_scope'] = 'retained event velocities; hash-checked full-run sample audit and thermo'
                if result['first_crossing']['step'] != retained['measured_step']:
                    raise ValueError('retained event is not first crossing')
    except (OSError, ValueError, UnicodeError, OverflowError, KeyError, TypeError) as exc:
        result.update(status='invalid_data', response_angle_deg=None, particle_identity_verified=False, reason=str(exc))
    result['source_sha256'] = source_sha256
    return result


def measure_rolling(post):
    """Read the thermo-selected protocol; never hide bad current data behind old dumps."""
    thermo = Path(post) / 'rolling_thermo.log'
    if thermo.exists():
        result = _measure_rolling_thermo(thermo)
    else:
        result = _measure_rolling_dumps(post)
    provenance = Path(post).parent.parent/'parameters.json'
    if provenance.is_file():
        try:
            protocol = json.loads(provenance.read_text(encoding='utf8')).get('rolling_protocol')
            if protocol is not None:
                if protocol.get('version') != result.get('protocol_version'):
                    raise ValueError('rolling protocol marker/provenance mismatch')
                if protocol.get('version') in ('V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
                    from optimizer.rolling_protocol import effective_protocol
                    current = effective_protocol(Path(post).parent, {'version':protocol['version']})
                    if current != protocol:
                        raise ValueError('rolling effective geometry/provenance mismatch')
                result['rolling_protocol'] = protocol
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            result.update(status='invalid_data',response_angle_deg=None,
                          particle_identity_verified=False,reason=str(exc))
    return result

