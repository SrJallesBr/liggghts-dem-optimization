"""Native rolling-event geometry. No polling, reconstruction or slope fitting."""
from pathlib import Path
import hashlib
import html
import math
import re
import subprocess
from optimizer.rolling_protocol import PRESENT_VERSION, fraction_denominator, population_fraction

from optimizer.rolling_measurement import read_dump, SPEED_THRESHOLD, LEGACY_SPEED_THRESHOLD, total_speed_moving
from optimizer.measurement_visuals import write_svg_png


def _moving(particle, metric):
    return (total_speed_moving(particle, metric.get('speed_threshold_m_s', LEGACY_SPEED_THRESHOLD))
            if metric.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION')
            else particle['vz'] < -LEGACY_SPEED_THRESHOLD)


def read_base_stl(path):
    """Read complete ASCII mesh/stl output, including every emitted triangle."""
    lines = [line.strip().split() for line in Path(path).read_text(encoding='ascii').splitlines() if line.strip()]
    triangles = []
    i = 0
    while i < len(lines):
        if lines[i][0] != 'solid': raise ValueError('base STL missing solid header')
        i += 1
        while i < len(lines) and lines[i][0] != 'endsolid':
            if i+6 >= len(lines): raise ValueError('truncated base STL')
            if lines[i][:2] != ['facet','normal'] or len(lines[i]) != 5:
                raise ValueError('malformed base STL facet')
            normal = list(map(float,lines[i][2:]))
            if not all(map(math.isfinite,normal)): raise ValueError('nonfinite STL normal')
            if lines[i+1] != ['outer','loop'] or lines[i+5] != ['endloop'] or lines[i+6] != ['endfacet']:
                raise ValueError('truncated or malformed base STL triangle')
            vertices = []
            for row in lines[i+2:i+5]:
                if row[0] != 'vertex' or len(row) != 4: raise ValueError('invalid STL vertex')
                v = tuple(map(float,row[1:]))
                if not all(map(math.isfinite,v)): raise ValueError('nonfinite STL vertex')
                vertices.append(v)
            triangles.append(vertices)
            i += 7
        if i == len(lines): raise ValueError('truncated base STL endsolid')
        i += 1
    if not triangles: raise ValueError('empty base STL')
    return triangles


def read_rolling_event(post, metric):
    """Cross-check the native event against FIRST crossing in the complete log.

    STL has no independent timestamp; native sequential writes and its resolved
    filename establish the clock. Hashes are provenance, not authenticity proof.
    """
    post = Path(post)
    if metric['status'] not in ('measured','censored'):
        raise ValueError('invalid full-run metric; event cannot be validated')
    first = metric.get('first_crossing')
    if first is None: raise ValueError('event exists without a threshold crossing')
    meta = post/'rolling_event.txt'
    raw = meta.read_text(encoding='ascii').strip()
    version = metric['protocol_version'] if metric.get('protocol_version') in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') else 'V1'
    match = re.fullmatch(r'ROLLING_EVENT_'+version+r',step=(\d+),angle=([^,\s]+),moving=(\d+),N0=(\d+),alive=(\d+)',raw)
    if not match: raise ValueError('malformed or extra event metadata')
    step, angle, moving, N0, alive = match.groups()
    step, moving, N0, alive = map(int,(step,moving,N0,alive))
    angle = float(angle)
    if not math.isfinite(angle) or not math.isclose(angle, first['angle_deg'], rel_tol=0.,abs_tol=1e-7):
        raise ValueError('event angle disagrees with prescribed gravity tilt')
    if any(value != first[key] for key,value in [('step',step),('moving',moving),('N0',N0),('alive',alive)]):
        raise ValueError('event is not the first sampled crossing or counts disagree')
    particle_path = post/'rolling_event_particles.dump'
    if metric.get('protocol_version') in ('V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION') and (post/'rolling_retention.json').exists():
        from optimizer.rolling_retention import read_retained_samples
        read_retained_samples(post)
        if not particle_path.exists(): particle_path = post/f'particles_{step}.dump'
    dump_step, particles, particle_sha = read_dump(particle_path,geometry=True)
    if dump_step != step or len(particles) != alive:
        raise ValueError('event particle clock/population mismatch')
    if sum(_moving(p, metric) for p in particles.values()) != moving:
        raise ValueError('event moving count disagrees with actual velocity threshold')
    if metric.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
        sample_step, sample_particles, _ = read_dump(post/f'particles_{step}.dump', geometry=True)
        if sample_step != step or particles != sample_particles:
            raise ValueError('event particles disagree with first-crossing particle dump')
    bases = list(post.glob('rolling_event_base*.stl'))
    if len(bases) != 1 or bases[0].name != f'rolling_event_base_{step}.stl':
        raise ValueError('event needs exactly one same-step emitted base STL')
    base_path = bases[0]
    triangles = read_base_stl(base_path)
    return dict(step=step,angle_deg=angle,moving=moving,N0=N0,alive=alive,lost=N0-alive,
                particles=list(particles.values()),triangles=triangles,
                paths=dict(particles=str(particle_path),base=str(base_path),metadata=str(meta)),
                source_sha256=dict(particles=particle_sha,base=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                                   metadata=hashlib.sha256(meta.read_bytes()).hexdigest()))


def rotate_zy(z, y, angle_deg):
    """Rotate about X by -angle, then project Z right and Y up.

    R_x(-angle) maps the deck gravity (0,-cos(angle),-sin(angle))
    to (0,-1,0). BOTH native particle centers and mesh vertices use this.
    """
    a = math.radians(-angle_deg)
    return math.cos(a)*z + math.sin(a)*y, -math.sin(a)*z + math.cos(a)*y


def write_rolling_svg(path, event, metric, trial_label=''):
    """Project ALL surviving spheres and emitted triangles with one m->px scale."""
    angle = event['angle_deg']
    grains = [(p,rotate_zy(p['z'],p['y'],angle)) for p in event['particles']]
    faces = [[rotate_zy(v[2],v[1],angle) for v in triangle] for triangle in event['triangles']]
    bounds = [(z-r,y-r,z+r,y+r) for p,(z,y) in grains for r in [p['radius']]]
    bounds += [(z,y,z,y) for triangle in faces for z,y in triangle]
    zmin,ymin = min(b[0] for b in bounds),min(b[1] for b in bounds)
    zmax,ymax = max(b[2] for b in bounds),max(b[3] for b in bounds)
    span_z,span_y = max(zmax-zmin,1e-9),max(ymax-ymin,1e-9)
    left,top,pw,ph = 80,210,1440,600
    # Reserve a separate detail panel only when the full base dwarfs the grains.
    grain_bounds = bounds[:len(grains)]
    gz0,gy0 = min(b[0] for b in grain_bounds),min(b[1] for b in grain_bounds)
    gz1,gy1 = max(b[2] for b in grain_bounds),max(b[3] for b in grain_bounds)
    detail_scale = min(360./max(gz1-gz0,1e-9),420./max(gy1-gy0,1e-9))
    full_scale = min((pw-120)/span_z,(ph-120)/span_y)
    show_detail = detail_scale > 1.5*full_scale
    if show_detail:
        pw = 980
    scale = min((pw-120)/span_z,(ph-120)/span_y)
    midz,midy = (zmin+zmax)/2,(ymin+ymax)/2
    X = lambda z: left+pw/2+(z-midz)*scale
    Y = lambda y: top+ph/2-(y-midy)*scale
    navy,teal,orange = '#263449','#14816b','#ef8c24'
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1060" viewBox="0 0 1600 1060">',
           '<rect width="1600" height="1060" fill="#f7f9fc"/>']
    def text(x,y,value,size=20,color=navy,weight='normal'):
        out.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="monospace" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(str(value))}</text>')
    text(80,55,'ROLAMENTO | PRIMEIRO SNAPSHOT ≥95%',34,weight='bold')
    text(80,95,f'{trial_label} | passo {event["step"]} | t={event["step"]*metric.get("dt_s", .0001):.3f} s | gravidade prescrita {angle:.2f}°',22)
    text(80,135,'vista equivalente; base fixa / gravidade rotacionada',22,teal)
    text(80,170,'Projeção lateral Z/Y · todas as partículas sobreviventes · eixos na mesma escala · rotação em X = −ângulo',18)
    out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" rx="14" fill="#f8fafc" stroke="#263449"/>')
    # Backdrop is an angular guide, never a pile-slope fit.
    # Register the guide to the actual fixed-base origin, not a screen corner.
    cx,cy = X(0.),Y(0.)
    pr = max(20., min(320., cx-left-28, left+pw-28-cx, cy-top-28))
    out.append(f'<circle data-kind="protractor-pivot" cx="{cx:.6f}" cy="{cy:.6f}" r="3" fill="{teal}"/>')
    out.append(f'<path d="M {cx-pr} {cy} A {pr} {pr} 0 0 1 {cx+pr} {cy}" fill="none" stroke="#dbe5ed" stroke-width="2"/>')
    for degree in range(0,181,5):
        a=math.radians(degree); length=18 if degree%15==0 else 9
        ux,uy=math.cos(a),-math.sin(a)
        out.append(f'<line data-kind="protractor-tick" x1="{cx+(pr-length)*ux:.5f}" y1="{cy+(pr-length)*uy:.5f}" x2="{cx+pr*ux:.5f}" y2="{cy+pr*uy:.5f}" stroke="#bfccd8"/>')
        if degree%15==0:
            text(cx+(pr+22)*ux-12,cy+(pr+22)*uy+5,f'{degree}°',14,'#78899a')
    a=math.radians(angle)
    out.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+pr*math.cos(a):.5f}" y2="{cy-pr*math.sin(a):.5f}" stroke="{teal}" stroke-opacity=".18" stroke-width="3"/>')
    for face in faces:
        coords=' '.join(f'{X(z):.6f},{Y(y):.6f}' for z,y in face)
        out.append(f'<polygon data-kind="base-triangle" points="{coords}" fill="{navy}" fill-opacity=".16" stroke="{navy}" stroke-opacity=".65" stroke-width="1.5"/>')
    for p,(z,y) in grains:
        moving=_moving(p, metric)
        out.append(f'<circle data-kind="particle" data-id="{int(p["id"])}" data-moving="{str(moving).lower()}" data-px-per-m="{scale:.12g}" cx="{X(z):.6f}" cy="{Y(y):.6f}" r="{p["radius"]*scale:.6f}" fill="{teal if moving else orange}" fill-opacity=".7" stroke="{navy}" stroke-width=".35"/>')
    # A metre reference makes equal-axis geometry auditable in exported figures.
    raw_length = 140./scale
    decade = 10.**math.floor(math.log10(raw_length))
    bar_m = max(v*decade for v in (1.,2.,5.,10.) if v*decade <= raw_length)
    bx,by = left+30,top+ph-27
    out.append(f'<line data-kind="scale-bar" data-length-m="{bar_m:.12g}" x1="{bx:.6f}" y1="{by}" x2="{bx+bar_m*scale:.6f}" y2="{by}" stroke="{navy}" stroke-width="3"/>')
    text(bx,by-10,f'{bar_m:g} m · mesma escala nos dois eixos',15,'#536477')
    out.append(f'<path d="M 135 255 V 315 l -7 -14 m 7 14 l 7 -14" fill="none" stroke="{navy}" stroke-width="3"/>')
    text(154,290,'g vertical',16)
    if show_detail:
        ix,iy,iw,ih = 1100,210,420,600
        out.append(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="14" fill="#f8fafc" stroke="#263449"/>')
        text(ix+20,iy+35,'DETALHE DOS GRÃOS',21,weight='bold')
        text(ix+20,iy+63,f'Ampliação {detail_scale/scale:.2f}× · mesma escala Z/Y',16,'#536477')
        out.append(f'<defs><clipPath id="rolling-grain-detail"><rect x="{ix+20}" y="{iy+85}" width="380" height="440"/></clipPath></defs>')
        IX = lambda z: ix+iw/2+(z-(gz0+gz1)/2)*detail_scale
        IY = lambda y: iy+305-(y-(gy0+gy1)/2)*detail_scale
        out.append('<g clip-path="url(#rolling-grain-detail)">')
        for face in faces:
            coords=' '.join(f'{IX(z):.6f},{IY(y):.6f}' for z,y in face)
            out.append(f'<polygon points="{coords}" fill="{navy}" fill-opacity=".12" stroke="{navy}" stroke-opacity=".35" stroke-width=".7"/>')
        for p,(z,y) in grains:
            moving=_moving(p, metric)
            out.append(f'<circle data-kind="inset-particle" data-id="{int(p["id"])}" data-px-per-m="{detail_scale:.12g}" cx="{IX(z):.6f}" cy="{IY(y):.6f}" r="{p["radius"]*detail_scale:.6f}" fill="{teal if moving else orange}" fill-opacity=".7" stroke="{navy}" stroke-width=".4"/>')
        out.append('</g>')
        raw_detail = 120./detail_scale
        ddecade = 10.**math.floor(math.log10(raw_detail))
        dbar = max(v*ddecade for v in (1.,2.,5.,10.) if v*ddecade <= raw_detail)
        out.append(f'<line data-kind="detail-scale-bar" data-length-m="{dbar:.12g}" x1="{ix+20}" y1="{iy+570}" x2="{ix+20+dbar*detail_scale:.6f}" y2="{iy+570}" stroke="{navy}" stroke-width="3"/>')
        text(ix+20,iy+558,f'{dbar:g} m · projeção com sobreposição',15,'#536477')
    version = metric.get('protocol_version')
    denominator = event['alive'] if version == PRESENT_VERSION else event['N0']
    label = 'presentes' if version == PRESENT_VERSION else 'N0 fixo'
    fraction = population_fraction(event['moving'], event['alive'], event['N0'], version)
    text(80,855,f'Snapshot: {event["moving"]}/{denominator} = {100*fraction:.2f}% ({label})  |  N0={event["N0"]}  alive={event["alive"]}  lost={event["lost"]}',23,weight='bold')
    if metric.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
        text(80,895,f'Métrica: primeiro cruzamento (1 amostra) | {metric["response_angle_deg"]:.2f}° | passo {metric["window_start_step"]}',22)
    elif metric['status'] == 'measured':
        text(80,895,f'Métrica (5 amostras): {metric["response_angle_deg"]:.2f}° | início {metric["window_start_step"]} | confirmação {metric["window_end_step"]}',22)
    else:
        text(80,895,'Métrica (5 amostras): censurada — sem janela válida até 45°; snapshot transitório preservado.',21)
    if metric.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
        text(80,932,f'Verde: velocidade total > {metric.get("speed_threshold_m_s", LEGACY_SPEED_THRESHOLD):.14g} m/s | Laranja: não atende ao critério.',19)
        text(80,967,'Movimento translacional coletivo; não comprova rolamento puro microscópico. Perdidas nunca contam como móveis.',18)
    else:
        text(80,932,f'Verde: vz < −{LEGACY_SPEED_THRESHOLD:.14f} m/s | Laranja: não atende ao critério (vz no referencial original).',19)
        text(80,967,'Movimento coletivo descendente; não comprova rolamento puro microscópico. Perdidas nunca contam como móveis.',18)
    text(80,1002,'Ângulo prescrito pela gravidade e pelo passo; nenhuma inclinação da pilha é inferida. Base: STL emitido no evento.',18)
    if metric.get('protocol_version') in ('V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION'):
        text(80,1035,f'Amostras: 0,1 s · parada no primeiro >=95%: passo {event["step"]}, {event["step"]*metric.get("dt_s", .0001):.1f} s · limite: 40 s / 90°',17,'#536477')
    else:
        text(80,1035,f'Amostras a cada 1000 passos (0,1 s) · execução mantida até {round((metric.get("tilt_start_s", 10.)+15.)/.0001)} / 45° · perdas no fim: {metric["lost_particle_count"]}',17,'#536477')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out),encoding='utf-8')


def render_rolling_snapshot(post, metric, *, expected=None):
    """Return explicit visualization status without changing the metric response."""
    post = Path(post)
    svg,png = post/'rolling_measurement.svg',post/'rolling_measurement.png'
    if (post/'rolling_retention.json').exists():
        try:
            data = read_rolling_event(post, metric)
            result = {k:v for k,v in data.items() if k not in ('particles','triangles')}
            result['paths'].update(svg=str(svg), png=str(png))
            return dict(result, status='rendered', reason=None)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return dict(status='visualization_failed', reason=str(exc), paths={})
    # A stale image must not represent a failed or non-crossing reprocessing.
    for old in (svg,png):
        old.unlink(missing_ok=True)
    first = metric.get('first_crossing')
    result = dict(status='visualization_failed',step=None,angle_deg=None,paths={},reason=None)
    if expected is None:
        deck=post.parent/'in.rolamento'
        expected = metric.get('native_event_protocol',False) or (
            deck.is_file() and any(f'print "ROLLING_EVENT_PROTOCOL_{v}"' in deck.read_text(encoding='utf-8')
                                   for v in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION')))
    exists = (post/'rolling_event.txt').exists() or (post/'rolling_event_particles.dump').exists() or any(post.glob('rolling_event_base*.stl'))
    if metric['status'] == 'invalid_data':
        result['reason']='invalid full-run thermo/metric; no visualization: '+str(metric.get('reason'))
        return result
    if not first:
        result.update(status='visualization_failed' if exists else 'not_reached',
                      reason='event files without a threshold crossing' if exists else ('never sampled >=95% of present particles (alive > 0)' if metric.get('protocol_version') == PRESENT_VERSION else 'never sampled >=95% of fixed N0'))
        return result
    if not exists and not expected:
        result.update(status='legacy_unavailable',reason='legacy/no-event directory; valid metric retained')
        return result
    result.update(expected_step=first['step'],expected_angle_deg=first['angle_deg'])
    try:
        data=read_rolling_event(post,metric)
        result.update({k:v for k,v in data.items() if k not in ('particles','triangles')})
        write_rolling_svg(svg,data,metric,post.parent.parent.name)
        result['paths']['svg']=str(svg)
        write_svg_png(svg,png)
        result['paths']['png']=str(png)
        result.update(status='rendered',reason=None)
    except (ValueError,OSError,UnicodeError,OverflowError,RuntimeError,subprocess.SubprocessError) as exc:
        png.unlink(missing_ok=True)
        result.update(status='visualization_failed',reason=f'{type(exc).__name__}: {exc}')
    return result
