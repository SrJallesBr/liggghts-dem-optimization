"""Read-only snapshot adapter and auditable exports; never launches a solver."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from optimizer.measurements import parse_ascii_stl_bounds, parse_vtk_points, parse_vtk_scalar_field
from optimizer.rocky_angles import measure_saor, measure_daor
from optimizer.visual_theme import measurement_theme, wrap_metadata


def _variables(deck):
    out = {}
    for line in deck.splitlines():
        match = re.match(r'\s*variable\s+(\w+)\s+(?:equal|index)\s+([^#]+)', line)
        if match:
            out[match[1]] = match[2].strip()
    return out


def evaluate(expression, variables, stack=()):
    expression = re.sub(r'\$\((.*?)\)', r'\1', expression)
    expression = re.sub(r'\$\{(\w+)\}', r'v_\1', expression)
    node = ast.parse(expression, mode='eval').body
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int,float):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id.startswith('v_'):
            key = n.id[2:]
            if key not in variables or key in stack:
                raise ValueError('unknown or cyclic deck variable: '+key)
            return evaluate(variables[key], variables, stack+(key,))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return visit(n.operand) * (-1 if isinstance(n.op,ast.USub) else 1)
        if isinstance(n, ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add): return a+b
            if isinstance(n.op,ast.Sub): return a-b
            if isinstance(n.op,ast.Mult): return a*b
            return a/b
        raise ValueError('unsupported arithmetic deck expression: '+expression)
    value=visit(node)
    if not math.isfinite(value): raise ValueError('nonfinite deck value')
    return value


def _mesh_geometry(case, deck, fix_name):
    """Read the declared trial-local mesh, never substitute a legacy filename."""
    declaration = re.search(r'(?m)^[ \t]*fix[ \t]+' + re.escape(fix_name)
                            + r'[ \t]+[^\n]+', deck.replace('&\n', ' '))
    if not declaration:
        raise ValueError('scaled mesh declaration missing: ' + fix_name)
    # Preserve quoted paths and immediate arithmetic, but never read options
    # from a comment (e.g. a commented-out scale or replacement filename).
    line = re.match(r'''(?:"[^"]*"|'[^']*'|[^#])*''', declaration[0])[0]
    tokens = re.findall(r'''"[^"]*"|'[^']*'|\$\([^)]*\)|\S+''', line)
    try:
        if tokens[3:5] != ['mesh/surface', 'file']:
            raise ValueError('expected mesh/surface file')
        mesh_file = tokens[5].strip('\"\'')
        scale_index = tokens.index('scale', 6)
        scale = evaluate(tokens[scale_index + 1].strip('\"\''), _variables(deck))
    except (IndexError, SyntaxError, ZeroDivisionError, ValueError) as exc:
        raise ValueError('invalid scaled mesh declaration: ' + fix_name) from exc
    path = (case / mesh_file).resolve()
    if not path.is_relative_to(case.resolve()):
        raise ValueError('mesh path escapes case directory')
    if scale <= 0:
        raise ValueError('positive mesh scale required')
    bounds = parse_ascii_stl_bounds(path.read_text(encoding='utf8'))
    bounds = {a: (b[0]*scale, b[1]*scale) for a, b in bounds.items()}
    if not all(math.isfinite(v) for pair in bounds.values() for v in pair):
        raise ValueError('finite scaled mesh bounds required')
    return bounds, {'mesh_file': mesh_file, 'mesh_path': str(path), 'mesh_scale': scale}


def _mesh_bounds(case, deck, fix_name):
    return _mesh_geometry(case, deck, fix_name)[0]


def geometry_for_snapshot(path,kind,points):
    case=Path(path).parents[1]
    name='in.deslizamento' if kind=='discharge' else 'in.repouso'
    deck=(case/name).read_text(encoding='utf8')
    variables=_variables(deck)
    ratio=evaluate('v_nbrratio',variables) if 'nbrratio' in variables else 1.
    if ratio<=0: raise ValueError('positive NBR ratio required')
    if kind=='discharge':
        bounds=_mesh_bounds(case,deck,'box')
        return {'bounds':bounds,'nbrratio':ratio,'source':'trial-local box STL and deck scale'}
    if '# ANSYS_REPOSE_10S_ONE_SHOT:' in deck:
        # New protocol: one tray. Its y=25 mm rim is NOT the floor.
        from optimizer.repose_ansys import read_triangles
        bounds=_mesh_bounds(case,deck,'base_repouso')
        match=re.search(r'(?m)^fix\s+base_repouso\s+\S+\s+mesh/surface\s+file\s+(\S+)',deck)
        tri=read_triangles(case/match[1],ratio)
        flat=tri[np.ptp(tri[:,:,1],axis=1)<1e-10]
        if not len(flat): raise ValueError('tray has no horizontal supporting facets')
        floor=float(flat[:,:,1].min())
        source='trial-local ANSYS tray STL; floor from actual horizontal facets'
    else:
        halves=[_mesh_bounds(case,deck,n) for n in ['base_repouso_esq','base_repouso_dir']]
        bounds={a:(min(b[a][0] for b in halves),max(b[a][1] for b in halves)) for a in ['x','y','z']}
        floor=bounds['y'][1]
        source='trial-local two tray STLs and deck scales; ROI top from saved particles'
    cx,cz=(sum(bounds['x'])/2,sum(bounds['z'])/2)
    half=min((bounds['x'][1]-bounds['x'][0])/2,(bounds['z'][1]-bounds['z'][0])/2)
    top=float(np.max(points[:,1]))
    if top<=floor: raise ValueError('no particles above tray')
    return {'center_xz':(cx,cz),'width':2*half,'vertical_bounds':(floor,top+max(top-floor,1e-9)*.01),
            'tray_half_width':half,'nbrratio':ratio,'tray_bounds':bounds,
            'source':source}


def json_ready(value):
    if isinstance(value,np.ndarray): return json_ready(value.tolist())
    if isinstance(value,np.generic): return json_ready(value.item())
    if isinstance(value,dict): return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_ready(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(json_ready(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def export_measurement(result,points,radii,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prefix=result['kind'];x=np.asarray(result['coordinates']);st=result['statistics']
    with (output/f'{prefix}_raw_data.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['coordinate_sim_m','minimum_max_y_sim_m','maximum_max_y_sim_m','mean_max_y_sim_m','occupied_samples'])
        for i in range(len(x)):w.writerow(json_ready([x[i],st['minimum'][i],st['maximum'][i],st['average'][i],st['support'][i]]))
    maxima=result['grid_maxima'];counts=result['grid_counts']
    with (output/f'{prefix}_cell_maxima.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['width_bin','sample_or_depth_bin','max_y_sim_m','particle_count'])
        for i in range(maxima.shape[0]):
            for j in range(maxima.shape[1]):w.writerow(json_ready([i,j,maxima[i,j],counts[i,j]]))
    with (output/f'{prefix}_treated_data.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['face','raw_bin','coordinate_sim_m','mean_max_y_sim_m','treatment','used_in_fit'])
        if prefix=='SAOR':
            fit_indices=set(result.get('fit_indices',[]))
            for i,reason in zip(np.flatnonzero(x<0),result['treatment_reasons']):
                w.writerow(json_ready(['symmetric_half',i,x[i],st['average'][i],reason,i in fit_indices]))
        else:
            for side,face in result['faces'].items():
                fit_indices=set(face.get('fit_indices',[]))
                for i,reason in zip(face.get('ordered_indices',[]),face.get('treatment_reasons',[])):
                    w.writerow(json_ready([side,i,x[i],st['average'][i],reason,i in fit_indices]))
    _render(result,points,radii,output/f'{prefix}.png')
    # Raw arrays are retained separately in CSV, as are all regression candidates in JSON.
    compact={k:v for k,v in result.items() if k not in ['grid_maxima','grid_counts']}
    write_json(output/f'{prefix}_Report.json',compact)
    from optimizer.rocky_audit import export_audit
    export_audit(result, points, radii, output)


@measurement_theme
def _render(result,points,radii,path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    p,r=np.asarray(points),np.asarray(radii)
    ratio=result['geometry']['nbrratio'];x=np.asarray(result['coordinates'])
    st=result['statistics'];avg=np.asarray(st['average']);low=np.asarray(st['minimum']);high=np.asarray(st['maximum'])
    kind=result['kind'];fig=Figure(figsize=(11,10),layout='constrained')
    FigureCanvasAgg(fig)
    axes=fig.subplots(2,1)
    ax=axes[0]
    ax.fill_between(x/ratio,low/ratio,high/ratio,color='#d5dce5',label='min–max dos máximos por célula')
    ax.plot(x/ratio,avg/ratio,'.-',color='#14966c',ms=4,lw=.7,label='perfil médio')
    fit_items=[]
    if kind=='SAOR' and 'fit' in result:fit_items=[('simétrico',result['fit'],result['fit_indices'])]
    if kind=='DAOR':fit_items=[(s,v['fit'],v['fit_indices']) for s,v in result['faces'].items() if v['status']=='measured']
    for side,fit,idx in fit_items:
        xx=x[idx]; yy=fit['slope']*xx+fit['intercept']
        ax.scatter(xx/ratio,avg[idx]/ratio,s=24,facecolors='none',edgecolors='#14966c',zorder=4)
        ax.plot(xx/ratio,yy/ratio,'--',lw=2,label=f'{side}: {fit["fit_angle_deg"]:.2f}°; R²={fit["r_squared"]:.4f}; n={fit["points_used"]}')
    response=result.get('response_angle_deg',result.get('mean_angle_deg'))
    title='sem ângulo mensurável' if response is None else f'resposta={response:.2f}°'
    ax.set_title(f'{kind} | {title} | {Path(result["source_path"]).parents[2].name}\nBaseado no manual Rocky — equivalência ANSYS não validada')
    ax.legend(fontsize=8);ax.grid();ax.set_ylabel('altura [m físicos, /NBR]')
    ax.set_xlabel('largura [m físicos, /NBR]');ax.set_aspect('equal',adjustable='box')
    ax=axes[1]
    if kind=='SAOR':
        cx,cz=result['geometry']['center_xz']; q=p.copy();q[:,0]-=cx;q[:,2]-=cz
        mask=np.abs(q[:,2])<=result['slab_width_m']/2;h=q[:,0]
        description='fatia 0°: partículas com raios físicos (não é a média das 36 orientações)'
    else:
        b=result['geometry']['bounds'];c=sum(b['x'])/2
        mask=(np.abs(p[:,0]-c)<=result['mean_diameter_m']/2)
        mask &= (p[:,1]>=b['y'][0]) & (p[:,1]<=b['y'][1]) & (p[:,2]>=b['z'][0]) & (p[:,2]<=b['z'][1])
        h=p[:,2]
        description='fatia central dentro da caixa: raios físicos (não é a média na profundidade)'
    circles=[Circle((h[i]/ratio,p[i,1]/ratio),r[i]/ratio) for i in np.flatnonzero(mask)]
    ax.add_collection(PatchCollection(circles,facecolor='#7d8da1',edgecolor='#536477',linewidth=.25))
    # A high flat top must not hide the bottom of a retained column. Include
    # the tray/container floor and complete radii of all displayed particles.
    floor=(result['floor_m'] if kind=='SAOR' else result['geometry']['bounds']['y'][0])/ratio
    xmin,xmax=axes[0].get_xlim();ymin,ymax=axes[0].get_ylim()
    ymin=min(ymin,floor)
    if mask.any():
        xmin=min(xmin,float(np.min(h[mask]-r[mask]))/ratio)
        xmax=max(xmax,float(np.max(h[mask]+r[mask]))/ratio)
        ymin=min(ymin,float(np.min(p[mask,1]-r[mask]))/ratio)
        ymax=max(ymax,float(np.max(p[mask,1]+r[mask]))/ratio)
    margin=max((ymax-ymin)*.02,1e-10)
    for panel in axes:
        panel.set_xlim(xmin,xmax);panel.set_ylim(ymin-margin,ymax+margin)
    ax.set_aspect('equal',adjustable='box');ax.set_title(description,fontsize=10)
    ax.set_xlabel('largura [m físicos, /NBR]');ax.set_ylabel('altura [m físicos, /NBR]');ax.grid()
    from optimizer.material_metadata import image_label
    fig.suptitle(wrap_metadata(image_label(result)+'\n'+str(result.get('reason') or 'Perfil observado')),
                 color='#b42318' if result['status']!='measured' else '#263449',fontsize=9)
    limits={'xlim':axes[1].get_xlim(),'ylim':axes[1].get_ylim(),'particles_shown':int(mask.sum())}
    fig.savefig(path,dpi=140)
    return limits


def measure_snapshot(path,kind,output,config=None):
    if kind not in ['discharge','repose']:raise ValueError('kind must be discharge or repose')
    path=Path(path).resolve();output=Path(output).resolve()
    case=path.parents[1]
    # Output in the original case is forbidden. Outputs beside a copied trial
    # are fine; the archived input tree remains read-only in offline usage.
    if output.is_relative_to(case):raise ValueError('output must be outside source case')
    payload=path.read_bytes();text=payload.decode('utf8')
    points=np.asarray(parse_vtk_points(text),float);radii=np.asarray(parse_vtk_scalar_field(text,'radius'),float)
    if points.size==0:raise ValueError('valid VTK with zero particles: angle unavailable')
    config=config or {};g=geometry_for_snapshot(path,kind,points)
    version=config.get('version','rocky-manual-2026r1-local-v1')
    modes={'rocky-manual-2026r1-local-v1':'strict_v1', 'mean-profile-local-v2':'mean_profile_v2'}
    if version not in modes: raise ValueError('unsupported measurement version: '+str(version))
    treatment_mode=modes[version]
    if kind=='repose':
        c=config.get('saor',{})
        result=measure_saor(points,radii,**{k:g[k] for k in ['center_xz','width','vertical_bounds','tray_half_width']},
                            divisions_factor=c.get('divisions_factor',1.5),number_of_samples=c.get('number_of_samples',36),
                            treatment_mode=treatment_mode,minimum_fit_r_squared=c.get('minimum_fit_r_squared',.95))
        result['response_angle_deg']=result['mean_angle_deg']
    else:
        c=config.get('daor',{})
        result=measure_daor(points,radii,bounds=g['bounds'],divisions_factor=c.get('divisions_factor',1.5),
                            response_convention=c.get('response_convention','left_face'),treatment_mode=treatment_mode,
                            minimum_fit_r_squared=c.get('minimum_fit_r_squared',.95))
    from optimizer.material_metadata import properties_from_deck
    result['material_properties'] = properties_from_deck(path, kind)
    if kind == 'repose' and config.get('saor', {}).get('retained_column_detection', False):
        from optimizer.response_policy import classify_retained_column
        tube, tube_source = _mesh_geometry(case, (case/'in.repouso').read_text(encoding='utf8'), 'tubo')
        tube_center = tuple(sum(tube[axis])/2 for axis in ['x','z'])
        tube_radius = min((tube[axis][1]-tube[axis][0])/2 for axis in ['x','z'])
        diagnosis = classify_retained_column(points, radii, center_xz=tube_center,
                                             tube_radius=tube_radius, floor=g['vertical_bounds'][0])
        diagnosis.update({'tube_' + key: value for key, value in tube_source.items()})
        diagnosis['tube_bounds_source'] = (
            f"trial-local {tube_source['mesh_file']} with deck scale {tube_source['mesh_scale']:.17g}; initial footprint")
        result['geometry_diagnosis'] = diagnosis
        # This snapshot heuristic cannot veto an otherwise valid finite face fit.
        # In particular a genuine vertical/near-vertical fit is still measured.
        valid_fit = (result['status'] == 'measured'
                     and isinstance(result.get('response_angle_deg'), (int,float))
                     and math.isfinite(result['response_angle_deg']))
        if diagnosis['detected'] and not valid_fit:
            result['estimator_before_geometry_gate'] = {
                key: result.get(key) for key in ['status','reason','mean_angle_deg','response_angle_deg','fit','candidate_fit']}
            result.update(status='unmeasurable', mean_angle_deg=None, response_angle_deg=None,
                          reason='Retained column: tall bulk confined to tube footprint with nearly constant width over height; no accepted repose face (local snapshot heuristic).')
            # The old fit remains available only as a diagnostic, never an accepted face.
            if 'fit' in result:
                result['candidate_fit'] = result.pop('fit')
                result.pop('fit_indices', None)
    if (kind == 'repose' and result['status'] == 'measured'
            and result.get('ninety_degree_rule_applied')):
        result['estimator_before_tray_convention'] = {
            key: result.get(key) for key in ['status','mean_angle_deg','response_angle_deg','fit']}
        result['candidate_fit'] = result.pop('fit')
        result.pop('fit_indices', None)
        result.update(status='unmeasurable', mean_angle_deg=None, response_angle_deg=None,
                      reason='Fitted repose toe outside tray: 90 deg comes from the tray convention, not a measured vertical angle.')
    result.update({'source_path':str(path),'source_sha256':hashlib.sha256(payload).hexdigest(),
                   'particle_count':len(points),'geometry':g,'config_used':config})
    export_measurement(result,points,radii,output)
    return result
