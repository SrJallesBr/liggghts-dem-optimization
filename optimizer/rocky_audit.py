"""Audit-only visualisations. Never modifies a measured profile or runs DEM."""
from __future__ import annotations
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from optimizer.rocky_io import write_json
from optimizer.visual_theme import measurement_theme, wrap_metadata


def export_audit(result, points, radii, output):
    """Write the exact DAOR spatial mesh, without interpolation or filling holes."""
    output=Path(output);kind=result['kind'];ratio=result['geometry']['nbrratio']
    if kind=='DAOR':
        maxima=np.asarray(result['grid_maxima']);counts=np.asarray(result['grid_counts'])
        horizontal_edges=np.asarray(result['horizontal_edges']);depth_edges=np.asarray(result['depth_edges'])
        role='measurement_grid';horizontal_axis,depth_axis='Z','X'
        faces=result['faces']
    else:
        from optimizer.rocky_angles import eulerian_maxima
        # A spatial XZ diagnostic is NOT the 36-rotation SAOR estimator.
        horizontal_edges=np.asarray(result['horizontal_edges']);depth_edges=horizontal_edges.copy()
        centred=np.asarray(points).copy()
        centred[:,0]-=result['center_xz'][0];centred[:,2]-=result['center_xz'][1]
        grid=eulerian_maxima(centred,horizontal_axis=0,depth_axis=2,
            horizontal_edges=horizontal_edges,depth_edges=depth_edges,
            vertical_bounds=(result['floor_m'],float(np.max(centred[:,1]))+1e-9))
        maxima,counts=grid['maxima'],grid['counts']
        role='spatial_diagnostic_not_saor_estimator';horizontal_axis,depth_axis='X','Z'
        faces={'symmetric_half':{**result,'ordered_indices':np.flatnonzero(result['coordinates']<0)}}
    he=horizontal_edges/ratio;de=depth_edges/ratio
    profile=[]
    for side,face in faces.items():
        used=set(face.get('fit_indices',[]));retained=set(face.get('treated_indices',[]))
        reasons=dict(zip(face.get('ordered_indices',[]),face.get('treatment_reasons',[])))
        centre=sum(result['bounds']['z'])/2 if kind=='DAOR' else 0.
        indices=np.flatnonzero(result['coordinates']<centre if side in ['left','symmetric_half'] else result['coordinates']>centre)
        for i in indices:
            state='fit' if i in used else 'retained' if i in retained else 'rejected'
            profile.append({'bin':int(i),'face':side,'state':state,'reason':reasons.get(i,'outside_treatment_start')})
    data={'schema':1,'kind':kind,'mesh_role':role,'horizontal_axis':horizontal_axis,'depth_axis':depth_axis,
          'horizontal_edges_physical_m':he,'depth_edges_physical_m':de,
          'max_y_physical_m':maxima/ratio,'particle_counts':counts,
          'empty_cells':int(np.isnan(maxima).sum()),'occupied_cells':int(np.isfinite(maxima).sum()),
          'interpolated':False,'measurement_changed':False,'nbrratio':ratio,
          'source_path':result['source_path'],'source_sha256':result['source_sha256'],
          'status':result['status'],'reason':result.get('reason'),'profile_bins':profile,
          'equivalence_to_ansys_verified':False}
    if kind=='SAOR':
        data.update(measurement_max_y_physical_m=result['grid_maxima']/ratio,
                    measurement_particle_counts=result['grid_counts'],sample_degrees=result['sample_degrees'])
    write_json(output/f'{kind}_mesh_audit.json',data)
    with (output/f'{kind}_mesh_cells.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['horizontal_bin','depth_bin','h_min_physical_m','h_max_physical_m','d_min_physical_m','d_max_physical_m','max_y_physical_m','particles'])
        for i,j in np.ndindex(maxima.shape):
            w.writerow([i,j,he[i],he[i+1],de[j],de[j+1],maxima[i,j]/ratio if np.isfinite(maxima[i,j]) else '',counts[i,j]])
    render_audit(result,data,output/f'{kind}_mesh_audit.png')
    names=[f'{kind}.png',f'{kind}_mesh_audit.png',f'{kind}_mesh_audit.json',f'{kind}_mesh_cells.csv',f'{kind}_Report.json',f'{kind}_raw_data.csv',f'{kind}_treated_data.csv',f'{kind}_cell_maxima.csv']
    manifest={'schema':1,'kind':kind,'status':result['status'],'reason':result.get('reason'),
              'source_path':result['source_path'],'source_sha256':result['source_sha256'],
              'artifacts':[{'path':name,'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()} for name in names]}
    # Ready marker is written last, after all images and hashes are complete.
    write_json(output/'Audit_Manifest.json',manifest)
    return data


@measurement_theme
def render_audit(result,data,path):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    heights=np.asarray(data['max_y_physical_m'],float);counts=np.asarray(data['particle_counts'])
    he=np.asarray(data['horizontal_edges_physical_m']);de=np.asarray(data['depth_edges_physical_m'])
    ratio=data['nbrratio'];finite=np.isfinite(heights)
    values=heights[finite];vmin=float(values.min()) if values.size else 0.;vmax=float(values.max()) if values.size else 1.
    norm=Normalize(vmin,vmax if vmax>vmin else vmin+1e-9)
    cmap=matplotlib.colormaps['viridis'].with_extremes(bad='#dedede')
    fig=Figure(figsize=(15,11),layout='constrained')
    FigureCanvasAgg(fig)
    a3=fig.add_subplot(221,projection='3d');top=fig.add_subplot(222);occ=fig.add_subplot(223);ax=fig.add_subplot(224)
    polygons=[];colors=[]
    for i,j in zip(*np.where(finite)):
        y=heights[i,j]
        polygons.append([(he[i],de[j],y),(he[i+1],de[j],y),(he[i+1],de[j+1],y),(he[i],de[j+1],y)])
        colors.append(cmap(norm(y)))
    if polygons:
        a3.add_collection3d(Poly3DCollection(polygons,facecolors=colors,edgecolors='#263449',linewidths=.25))
    floor=(result['bounds']['y'][0] if result['kind']=='DAOR' else result['floor_m'])/ratio
    a3.set_xlim(he[0],he[-1]);a3.set_ylim(de[0],de[-1]);a3.set_zlim(min(floor,vmin),max(vmax,floor+1e-9))
    spans=[he[-1]-he[0],de[-1]-de[0],max(vmax-floor,1e-9)]
    a3.set_box_aspect(spans);a3.view_init(elev=28,azim=-57)
    a3.set_xlabel(f"{data['horizontal_axis']} [m físicos]");a3.set_ylabel(f"{data['depth_axis']} [m físicos]");a3.set_zlabel('máximo Y [m físicos]')
    a3.set_title('Malha 2,5D: patamares por célula\nSem interpolar nem ligar células vazias',fontsize=10)
    mesh=top.pcolormesh(he,de,np.ma.masked_invalid(heights.T),cmap=cmap,norm=norm,edgecolors='#263449',linewidth=.25,shading='flat')
    fig.colorbar(mesh,ax=top,label='máximo Y dos centros [m físicos]')
    top.set_title(f"Vista superior | {data['occupied_cells']} ocupadas / {data['empty_cells']} vazias (cinza)",fontsize=10)
    top.set_aspect('equal',adjustable='box');top.set_xlabel(f"{data['horizontal_axis']} [m físicos]");top.set_ylabel(f"{data['depth_axis']} [m físicos]")
    if result['kind']=='DAOR':
        mesh=occ.pcolormesh(he,de,counts.T,cmap='magma',edgecolors='#263449',linewidth=.25,shading='flat')
        fig.colorbar(mesh,ax=occ,label='número de centros por célula')
        occ.set_title('Ocupação: zero não é altura zero',fontsize=10);occ.set_aspect('equal',adjustable='box')
        occ.set_xlabel(f"{data['horizontal_axis']} [m físicos]");occ.set_ylabel(f"{data['depth_axis']} [m físicos]")
    else:
        samples=np.asarray(data['measurement_max_y_physical_m'])
        mesh=occ.pcolormesh(he,np.arange(37)*10,np.ma.masked_invalid(samples.T),cmap=cmap,edgecolors='#263449',linewidth=.15,shading='flat')
        fig.colorbar(mesh,ax=occ,label='máximo Y por fatia/orientação [m físicos]')
        occ.set_title('Medição SAOR: 36 orientações, NÃO células vizinhas no espaço',fontsize=9)
        occ.set_xlabel('posição na fatia rotacionada [m físicos]');occ.set_ylabel('orientação [graus]')
        a3.set_title('Malha XZ 2,5D DIAGNÓSTICA\nNão substitui as 36 orientações SAOR',fontsize=10)
        top.set_title(f"Diagnóstico XZ | {data['occupied_cells']} ocupadas / {data['empty_cells']} vazias (cinza)",fontsize=10)
    x=np.asarray(result['coordinates'])/ratio;st=result['statistics'];avg=np.asarray(st['average'])/ratio
    ax.fill_between(x,np.asarray(st['minimum'])/ratio,np.asarray(st['maximum'])/ratio,color='#d9e1e8',label='min–max dos máximos')
    ax.plot(x,avg,color='#7d8da1',lw=.7,label='média sem preencher vazios')
    for state,color,marker,label in [('rejected','#d93535','x','fora do trecho/rejeitado'),('retained','#ef8c24','o','retido; fora da reta escolhida'),('fit','#14966c','o','usado na regressão')]:
        ids=sorted({p['bin'] for p in data['profile_bins'] if p['state']==state})
        if ids: ax.scatter(x[ids],avg[ids],s=28,c=color,marker=marker,label=label,zorder=4)
    fits=([(s,f['fit'],f['fit_indices']) for s,f in result['faces'].items() if f['status']=='measured']
          if result['kind']=='DAOR' else [('simétrico',result['fit'],result['fit_indices'])] if 'fit' in result else [])
    for side,fit,ids in fits:
        xx=np.asarray(result['coordinates'])[ids]
        ax.plot(xx/ratio,(xx*fit['slope']+fit['intercept'])/ratio,'--',label=f"{side}: {fit['fit_angle_deg']:.2f}° R²={fit['r_squared']:.4f} n={fit['points_used']}")
    response=result.get('response_angle_deg',result.get('mean_angle_deg'))
    ax.set_title('Perfil e seleção | '+('SEM ÂNGULO' if response is None else f'resposta {response:.2f}°'),fontsize=10)
    ax.set_xlabel('largura [m físicos]');ax.set_ylabel('altura Y [m físicos]');ax.set_aspect('equal',adjustable='box');ax.grid();ax.legend(fontsize=7)
    trial=Path(result['source_path']).parents[2].name
    from optimizer.material_metadata import image_label
    fig.suptitle(wrap_metadata(f"{result['kind']} | {trial} | {result['status']}\nMáximos dos CENTROS; equivalência ANSYS não validada\n"+image_label(result)),fontsize=10)
    fig.savefig(path,dpi=140)


def write_gallery(output):
    """Self-contained local index; uses reports, not timestamps or guessed angles."""
    output=Path(output);sections=[]
    for report in sorted(output.glob('*/Pair_Report.json')):
        pair=json.loads(report.read_text('utf8'));items=[]
        for kind,response in pair['responses'].items():
            prefix='SAOR' if kind=='repose' else 'DAOR'
            directory=report.parent/kind
            text=f"<h3>{html.escape(kind)} — {html.escape(response['status'])}</h3><p>{html.escape(str(response.get('reason') or 'Medido; não é validação física'))}</p>"
            if kind == 'rolling':
                qc=directory/'Rolling_Report.json'
                text+='<p>Movimento coletivo descendente; não comprova rolamento microscópico puro.</p>'
                if qc.is_file():
                    url=html.escape(qc.relative_to(output).as_posix(),quote=True)
                    text+=f'<p><a href="{url}">Rolling_Report.json</a></p>'
                items.append(text)
                continue
            manifest=directory/'Audit_Manifest.json'
            if manifest.is_file():
                records=json.loads(manifest.read_text('utf8'))['artifacts']
                for artifact in records:
                    path=directory/artifact['path']
                    if not path.resolve().is_relative_to(directory.resolve()): raise ValueError('artifact escapes audit directory')
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']: raise ValueError('audit artifact hash mismatch')
                    relative=path.relative_to(output).as_posix();url=html.escape(relative,quote=True)
                    if path.name in [prefix+'.png',prefix+'_mesh_audit.png']:
                        text+=f'<a href="{url}"><img loading="lazy" src="{url}" alt="{prefix} {html.escape(response["status"])}"></a>'
                    text+=f'<p><a href="{url}">{html.escape(path.name)}</a></p>'
            else:
                text+='<p>Sem imagem: snapshot ausente, inválido ou exportação incompleta.</p>'
            items.append(text)
        link=html.escape(report.relative_to(output).as_posix(),quote=True)
        sections.append(f'<section><h2>{html.escape(report.parent.name)}</h2><p>{html.escape(pair["source_workspace"])}</p><a href="{link}">Pair_Report.json</a>'+''.join(items)+'</section>')
    header='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Auditoria DEM 2,5D</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f3f5f7;color:#142334}section{background:white;padding:20px;margin:20px 0}img{width:49%;vertical-align:top}p{overflow-wrap:anywhere}</style><h1>Auditoria DEM 2,5D</h1><p>Dados salvos; nenhuma simulação nova. Vazios não preenchidos. Equivalência ANSYS não validada. No SAOR, a malha XZ é diagnóstico; o estimador continua usando as 36 orientações.</p>'
    (output/'index.html').write_text(header+''.join(sections)+'</html>',encoding='utf8')
