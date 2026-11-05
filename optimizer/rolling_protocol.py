"""Auditable native event grammar. Includes are data, never executed by Python.

Deliberately accept only this small protocol, not arbitrary nested LIGGGHTS code.
Keep these command sequences and the shipped .inc file in sync.
"""
import re
import math
import struct
import hashlib
from pathlib import Path

HEIGHT_VERSION = 'V6_LIFT_HEIGHT_3S'
PRESENT_VERSION = 'V7_PRESENT_POPULATION'


def fraction_denominator(version):
    return 'present_population' if version == PRESENT_VERSION else 'initial_population'


def population_fraction(moving, alive, initial, version):
    denominator = alive if version == PRESENT_VERSION else initial
    return moving / denominator if denominator > 0 else 0.


def threshold_crossed(moving, alive, initial, version):
    denominator = alive if version == PRESENT_VERSION else initial
    return alive > 0 and denominator > 0 and 20*moving >= 19*denominator


def mesh_y_bounds(path):
    """Validate an entire binary/ASCII STL before using its SI y extent."""
    data = Path(path).read_bytes()
    vertices = []
    if len(data) >= 84 and len(data) == 84 + 50*struct.unpack_from('<I', data, 80)[0]:
        for normal_and_vertices in struct.iter_unpack('<12fH', data[84:]):
            if not all(math.isfinite(v) for v in normal_and_vertices[:12]):
                raise ValueError('nonfinite binary STL')
            vertices.extend(normal_and_vertices[i:i+3] for i in (3,6,9))
    else:
        try:
            text = data.decode('ascii').strip()
        except UnicodeError as exc:
            raise ValueError('malformed binary/ASCII STL') from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 9 or not lines[0].startswith('solid') or not lines[-1].startswith('endsolid'):
            raise ValueError('malformed ASCII STL')
        body = lines[1:-1]
        if len(body) % 7:
            raise ValueError('incomplete ASCII STL facets')
        for offset in range(0, len(body), 7):
            facet = body[offset:offset+7]
            if (not facet[0].startswith('facet normal ') or facet[1] != 'outer loop'
                    or facet[5:] != ['endloop','endfacet']):
                raise ValueError('malformed ASCII STL facet')
            try:
                normal = list(map(float, facet[0].split()[2:]))
                points = [list(map(float, line.split()[1:])) for line in facet[2:5]]
                if (len(normal) != 3 or any(not line.startswith('vertex ') for line in facet[2:5])
                        or any(len(p) != 3 for p in points)
                        or not all(math.isfinite(v) for p in [normal]+points for v in p)):
                    raise ValueError('invalid ASCII STL coordinates')
                vertices.extend(points)
            except (ValueError, OverflowError) as exc:
                raise ValueError('invalid ASCII STL coordinates') from exc
    if not vertices:
        raise ValueError('empty STL')
    lo, hi = min(p[1] for p in vertices), max(p[1] for p in vertices)
    if hi <= lo:
        raise ValueError('STL must have positive finite y height')
    return lo, hi


def normalize_protocol(config):
    version = config.get('version', 'V5')
    if version not in ('V5', HEIGHT_VERSION, PRESENT_VERSION):
        raise ValueError('unsupported rolling protocol')
    expected = dict(version=version, speed_threshold_m_s=.001, fraction_threshold=.95,
                    consecutive_samples=1, retention='validated-event-image-v1',
                    settle_s=[1,3], lift_s=[3,6] if version in (HEIGHT_VERSION, PRESENT_VERSION) else [3,10],
                    tilt_s=[10,40], tilt_rate_deg_s=3., maximum_angle_deg=90., early_stop=True)
    if version in (HEIGHT_VERSION, PRESENT_VERSION):
        expected.update(rest_s=[6,10], lift_speed_m_s='mesh_height/3s')
    else:
        expected['lift_speed_m_s'] = .025
    if version == PRESENT_VERSION:
        expected['fraction_denominator'] = fraction_denominator(version)
    for key, value in config.items():
        if key not in expected or value != expected[key] or (isinstance(value, bool) != isinstance(expected[key], bool)):
            raise ValueError('inconsistent rolling protocol configuration: '+key)
    return expected


def effective_protocol(folder, config):
    protocol = normalize_protocol(config)
    if protocol['version'] == 'V5':
        return protocol
    from optimizer.calibration_controller import CalibrationController
    folder = Path(folder)
    deck = folder/'in.rolamento'
    text = deck.read_text(encoding='utf8')
    active = '\n'.join(line.split('#',1)[0].strip() for line in text.splitlines())
    if len(re.findall(r'^fix copo all mesh/surface file meshes/copo.stl type 2 scale \$\(v_nbrratio\)$', active, re.M)) != 1:
        raise ValueError('unsupported copo mesh transform/scale')
    for name in ('dt','radius1','radiusnbr','nbrratio','angulo_gravidade'):
        if len(re.findall(r'^variable\s+'+name+r'\s+', active, re.M)) != 1:
            raise ValueError('ambiguous rolling protocol variable: '+name)
    match = re.search(r'^variable dt equal (5e-5|0\.00005|1e-4|0\.0001)\s*$', active, re.M)
    if not match:
        raise ValueError('unsupported rolling timestep')
    dt = float(match.group(1))
    gravity = ('(step>=200000)*(step-200000)*0.000002617993877991495' if dt == 5e-5 else
               '(step>=100000)*(step-100000)*0.00000523598775598299')
    if 'timestep ${dt}' not in active or 'variable angulo_gravidade equal '+gravity not in active:
        raise ValueError('rolling gravity timing/rate changed')
    scale = CalibrationController._nbr_ratio_from_deck(deck)
    fill = folder/'in.fill_1s'
    if fill.exists():
        fill_active = '\n'.join(line.split('#',1)[0].strip() for line in fill.read_text(encoding='utf8').splitlines())
        fill_match = re.search(r'^variable dt equal (5e-5|0\.00005|1e-4|0\.0001)\s*$', fill_active, re.M)
        if not fill_match or float(fill_match.group(1)) != dt:
            raise ValueError('rolling fill dt mismatch')
        for name in ('dt','radius1','radiusnbr','nbrratio'):
            if len(re.findall(r'^variable\s+'+name+r'\s+', fill_active, re.M)) != 1:
                raise ValueError('ambiguous rolling fill variable: '+name)
        if CalibrationController._nbr_ratio_from_deck(fill) != scale:
            raise ValueError('rolling fill/main scale mismatch')
    mesh = folder/'meshes/copo.stl'
    low, high = mesh_y_bounds(mesh)
    height = (high-low)*scale
    if not math.isfinite(height) or height <= 0:
        raise ValueError('invalid scaled mesh height')
    protocol.update(dt_s=dt, mesh_y_bounds_m=[low,high], mesh_scale=scale, effective_height_m=height,
                    lift_speed_m_s=height/3., mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    return protocol

EVENT_INCLUDES = {
    'rolling_event_check.inc': '''if "${rolling_event_saved} == 0 && ${rolling_event_crossed} == 1" then &
  "write_dump all custom post/rolling_event_particles.dump id type x y z radius vx vy vz omegax omegay omegaz modify sort id format '%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g'" &
  "write_dump all mesh/stl post/rolling_event_base_*.stl base" &
  "print ROLLING_EVENT_V5,step=$(step),angle=${angulo_gravidade_graus},moving=$(c_moving_count),N0=${initial_count},alive=$(count(all)) file post/rolling_event.txt screen no" &
  "variable rolling_event_saved equal 1"''',
}

TILT_COMMANDS = '''log post/rolling_thermo.log
variable moving_flag atom sqrt(vx*vx+vy*vy+vz*vz)>0.001
compute moving_count all reduce sum v_moving_flag
variable initial_count equal $(count(all))
variable rolling_event_saved equal 0
variable rolling_event_crossed equal (step>=200000)*(20*c_moving_count>=19*v_initial_count)
variable dumptime equal 0.1
variable dumpstep equal round(v_dumptime/v_dt)
thermo ${dumpstep}
thermo_style custom step time atoms v_angulo_gravidade_graus c_moving_count v_initial_count
thermo_modify lost ignore flush yes format float %.17g
dump rolling_particles all custom 2000 post/particles_*.dump id type x y z radius vx vy vz omegax omegay omegaz
dump_modify rolling_particles sort id first yes format "%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g"
print "ROLLING_EVENT_PROTOCOL_V5"
print "SETTLE_1_TO_3S"
run 0
run 40000 pre no post yes
print "LIFT_MOLD_3_TO_10S_AT_0P025M_S"
fix lift_copo all move/mesh mesh copo linear 0.0 0.025 0.0
run 140000 pre yes post yes
unfix lift_copo
print "GRAVITY_ROTATION_10_TO_40S_3DEG_S_90DEG"
run 0
include rolling_event_check.inc
if "${rolling_event_saved} == 1" then "jump SELF rolling_done"
variable rolling_tilt_sample loop 300
label rolling_tilt_loop
run 2000 pre no post yes
include rolling_event_check.inc
if "${rolling_event_saved} == 1" then "jump SELF rolling_done"
next rolling_tilt_sample
jump SELF rolling_tilt_loop
label rolling_done
print "ROLLING_THERMO_COMPLETE"'''


def commands(text):
    result, pending = [], ''
    for line in text.splitlines():
        line = line.split('#',1)[0].strip()
        if not line: continue
        if line.endswith('&'):
            pending += line[:-1].rstrip()+' '
        else:
            result.append(pending+line)
            pending = ''
    if pending: result.append(pending)  # truncated continuation cannot match
    return result


def tilt_commands(protocol):
    if protocol['version'] == 'V5':
        return TILT_COMMANDS
    if protocol['version'] not in (HEIGHT_VERSION, PRESENT_VERSION):
        raise ValueError('unsupported rolling protocol')
    speed = protocol['lift_speed_m_s']
    if not isinstance(speed, (int,float)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError('effective mesh lift speed required')
    commands_text = TILT_COMMANDS
    if protocol['version'] == PRESENT_VERSION:
        commands_text = commands_text.replace('(20*c_moving_count>=19*v_initial_count)',
            '(count(all)>0)*(20*c_moving_count>=19*count(all))')
    commands_text = (commands_text.replace('ROLLING_EVENT_PROTOCOL_V5', 'ROLLING_EVENT_PROTOCOL_'+protocol['version'])
            .replace('LIFT_MOLD_3_TO_10S_AT_0P025M_S', 'LIFT_MOLD_FULL_HEIGHT_3_TO_6S')
            .replace('linear 0.0 0.025 0.0', f'linear 0.0 {speed:.17g} 0.0')
            .replace('run 140000 pre yes post yes\nunfix lift_copo',
                     'run 60000 pre yes post yes\nunfix lift_copo\nprint "REST_6_TO_10S"\nrun 80000 pre yes post yes'))
    if protocol.get('dt_s', 5e-5) == 1e-4:
        for old, new in [('step>=200000', 'step>=100000'),
                         ('custom 2000 post/particles_', 'custom 1000 post/particles_'),
                         ('run 40000 pre no post yes', 'run 20000 pre no post yes'),
                         ('run 60000 pre yes post yes', 'run 30000 pre yes post yes'),
                         ('run 80000 pre yes post yes', 'run 40000 pre yes post yes'),
                         ('run 2000 pre no post yes', 'run 1000 pre no post yes')]:
            if commands_text.count(old) != 1:
                raise ValueError('unexpected rolling command grammar')
            commands_text = commands_text.replace(old, new)
    return commands_text


def event_includes(version):
    return {name: text.replace('ROLLING_EVENT_V5', 'ROLLING_EVENT_'+version)
            for name, text in EVENT_INCLUDES.items()}


def validate_native_event(folder, text):
    """Fail closed on unreviewed include/run/dump paths or population commands."""
    active = commands(text)
    try:
        start = active.index('log post/rolling_thermo.log')
    except ValueError:
        raise ValueError('rolling event protocol missing thermo log')
    markers = re.findall(r'^\s*print "ROLLING_EVENT_PROTOCOL_([^"]+)"', text, re.M)
    if len(markers) != 1 or markers[0] not in ('V5', HEIGHT_VERSION, PRESENT_VERSION):
        raise ValueError('unknown or ambiguous rolling protocol marker')
    version = markers[0]
    protocol = effective_protocol(folder, {'version':version})
    if active[start:] != commands(tilt_commands(protocol)):
        raise ValueError('rolling event protocol/timing or population commands changed')
    for name, expected in event_includes(version).items():
        path = folder/name
        if not path.is_file() or commands(path.read_text(encoding='utf-8')) != commands(expected):
            raise ValueError('rolling event include missing or unapproved commands: '+name)
    for line in active[:start]:
        if re.match(r'(?:include|if|jump|next|label|python|run|dump|write_dump|delete_atoms|create_atoms|read_data|replicate|clear)\b', line):
            raise ValueError('rolling event unexpected execution/population command')
        if line.startswith('shell ') and line != 'shell "mkdir -p post"':
            raise ValueError('rolling event unexpected shell command')
        if line.startswith('fix ') and line.split()[3] not in {
                'nve/sphere','property/global','mesh/surface/stress','mesh/surface','addforce','wall/gran'}:
            raise ValueError('rolling event unapproved fix/population command')
