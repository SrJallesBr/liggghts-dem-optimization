"""Future-only ANSYS geometry repose, one finite-resolution PSD-constrained fill.

No prescribed mass/packing fraction. A seeded ordering of a Cartesian candidate
net (spacing r_min/2) spans the usable cylinder. Choose the next PSD class by
largest number-fraction deficit (mass fractions converted by r**3), then accept
the first candidate with full face/base/top/pair clearance. When a class no
longer fits, choose the largest deficit among feasible classes, until even the
smallest radius has no candidate. Finite-grid saturation is independently
audited for all seven radii; NOT a proof of continuum densest packing. All particles exist at t=0; no refill.
"""
from pathlib import Path
import hashlib,json,struct
import numpy as np
from scipy.spatial import cKDTree
RADII=np.array([.250,.275,.325,.375,.425,.475,.525])/2
FRACTIONS=np.array([.930,.030,.005,.005,.010,.005,.015])
RATIO=RADII[0]/.01
DENSITY=5280.
BASE=0.0 # actual horizontal tray facets; 25 mm is rim height, not floor
TOP=.650*RATIO

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_triangles(path,ratio=RATIO):
 data=Path(path).read_bytes()
 if len(data)>=84 and 84+50*struct.unpack_from('<I',data,80)[0]==len(data):
  triangles=[struct.unpack_from('<12fH',data,84+50*i)[3:12] for i in range(struct.unpack_from('<I',data,80)[0])]
 else:
  vertices=[list(map(float,line.split()[1:])) for line in data.decode().splitlines() if line.strip().startswith('vertex ')]
  triangles=np.array(vertices).reshape(-1,9)
 return np.array(triangles).reshape(-1,3,3)*(.001*ratio)

def minimum_triangle_clearance(p,triangles):
 best=np.full(len(p),np.inf); q=p[:,:3]
 for a,b,c in triangles:
  ab=b-a; ac=c-a; normal=np.cross(ab,ac); norm=np.linalg.norm(normal)
  if norm<1e-15: continue
  normal/=norm; signed=(q-a)@normal; projected=q-signed[:,None]*normal
  v=projected-a; d00=ab@ab; d01=ab@ac; d11=ac@ac
  u=((v@ab)*d11-(v@ac)*d01)/(d00*d11-d01*d01)
  w=((v@ac)*d00-(v@ab)*d01)/(d00*d11-d01*d01)
  face=np.where((u>=0)&(w>=0)&(u+w<=1),abs(signed),np.inf)
  for start,end in ((a,b),(b,c),(c,a)):
   edge=end-start; t=np.clip((q-start)@edge/(edge@edge),0,1)
   face=np.minimum(face,np.linalg.norm(q-start-t[:,None]*edge,axis=1))
  best=np.minimum(best,face)
 return float((best-p[:,3]).min())

def geometry_planes(path,ratio=RATIO):
 planes={}
 for t in read_triangles(path,ratio):
  normal=np.cross(t[1]-t[0],t[2]-t[0]); norm=np.linalg.norm(normal)
  if norm<1e-12: continue
  normal/=norm
  if abs(normal[1])>1e-6: continue
  nxz=normal[[0,2]]; b=float(nxz@t[0,[0,2]])
  if b<0: nxz=-nxz; b=-b
  key=tuple(np.round(np.r_[nxz,b],9)); planes[key]=np.r_[nxz,b]
 return np.array(list(planes.values()))

def candidate_net(planes):
 step=RADII[0]/2
 axis=np.arange(-1.4,1.4+step/2,step)
 levels=np.arange(BASE+RADII[0]+1e-8,TOP-RADII[0],step)
 x,z=np.meshgrid(axis,axis,indexing='ij'); xz=np.c_[x.ravel(),z.ravel()]
 wall=(planes[:,2,None]-planes[:,:2]@xz.T).min(axis=0)
 xz=xz[wall>=RADII[0]]; wall=wall[wall>=RADII[0]]
 pos=np.array([[x,y,z] for (x,z) in xz for y in levels],dtype=float)
 clear=np.minimum(np.repeat(wall,len(levels)),np.minimum(pos[:,1]-BASE,TOP-pos[:,1]))
 return pos,clear

def build_particles(planes,seed=20261104):
 pos,clear=candidate_net(planes)
 rng=np.random.default_rng(seed); order=rng.permutation(len(pos)); pos=pos[order]; clear=clear[order]
 weights=FRACTIONS/RADII**3; weights/=weights.sum()
 counts=np.zeros(7,dtype=int); particles=[]
 while True:
  feasible=RADII+1e-9<=float(clear.max())
  if not feasible.any(): break
  k=int(np.argmax(np.where(feasible,weights*(sum(counts)+1)-counts,-np.inf)))
  valid=np.flatnonzero(clear>=RADII[k]+1e-9)
  center=pos[valid[0]].copy(); particles.append([*center,RADII[k]])
  counts[k]+=1
  clear=np.minimum(clear,np.linalg.norm(pos-center,axis=1)-RADII[k])
 return np.array(particles)

def audit_particles(p,planes):
 wall=(planes[:,2,None]-planes[:,:2]@p[:,[0,2]].T-p[:,3]).min()
 pairs=cKDTree(p[:,:3]).query_pairs(2*max(RADII)+1e-8,output_type='ndarray')
 clearance=np.linalg.norm(p[pairs[:,0],:3]-p[pairs[:,1],:3],axis=1)-p[pairs[:,0],3]-p[pairs[:,1],3]
 counts=np.array([sum(np.isclose(p[:,3],r)) for r in RADII]); masses=counts*RADII**3
 actual=masses/masses.sum()
 next_class=int(np.argmax((FRACTIONS/RADII**3)/(FRACTIONS/RADII**3).sum()*(len(p)+1)-counts))
 bins=np.linspace(BASE,TOP,11); bin_counts=np.histogram(p[:,1],bins)[0]
 candidates,remaining=candidate_net(planes)
 distances=cKDTree(candidates).sparse_distance_matrix(cKDTree(p[:,:3]),2*max(RADII)+1e-8,output_type='coo_matrix')
 np.minimum.at(remaining,distances.row,distances.data-p[distances.col,3])
 residual=[int(np.sum(remaining>=r+1e-9)) for r in RADII]
 return dict(remaining_candidate_count_by_radius=residual,height_bin_edges_m=bins.tolist(),height_bin_particle_counts=bin_counts.tolist(),algorithm='seeded PSD-deficit Cartesian candidate net; smallest-radius grid saturation',seed=20261104,candidate_spacing_m=float(RADII[0]/2),candidate_grid_exhausted_for_next_psd_class=not any(residual),next_blocked_radius_m=float(RADII[next_class]),particle_count=len(p),radius_m=RADII.tolist(),radius_counts=counts.tolist(),expected_mass_fractions=FRACTIONS.tolist(),actual_mass_fractions=actual.tolist(),max_absolute_mass_fraction_error=float(abs(actual-FRACTIONS).max()),density_kg_m3=DENSITY,total_mass_kg=float((4*np.pi/3*DENSITY*masses).sum()),minimum_pair_clearance_m=float(clearance.min()),minimum_wall_clearance_m=float(wall),minimum_base_clearance_m=float((p[:,1]-p[:,3]-BASE).min()),minimum_top_clearance_m=float((TOP-p[:,1]-p[:,3]).min()),bounds_m=[[-7.8,7.8],[-.3,16.55],[-7.8,7.8]],no_replenishment=True,settled_or_stable_claim=False)

def write_population(case):
 case=Path(case); planes=geometry_planes(case/'meshes/Container_Cylinder.stl')
 p=build_particles(planes); audit=audit_particles(p,planes)
 if min(audit[k] for k in ('minimum_pair_clearance_m','minimum_wall_clearance_m','minimum_base_clearance_m','minimum_top_clearance_m')) < -1e-10: raise ValueError('initial overlap')
 header=f'LIGGGHTS one-shot geometry fill\n\n{len(p)} atoms\n4 atom types\n\n-7.8 7.8 xlo xhi\n-0.3 16.55 ylo yhi\n-7.8 7.8 zlo zhi\n\nAtoms # granular\n\n'
 rows=['%d 1 %.17g %.17g %.17g %.17g %.17g'%(i,2*r,DENSITY,x,y,z) for i,(x,y,z,r) in enumerate(p,1)]
 (case/'initial_particles.data').write_text(header+'\n'.join(rows)+'\n')
 audit['initial_data_sha256']=sha(case/'initial_particles.data')
 (case/'initial_audit.json').write_text(json.dumps(audit,indent=2))
 return audit

def require_scientific_protocol_compatibility(root, config=None):
 """Deny campaign solver launch; isolated baseline validation is not DOE readiness."""
 path=Path(root)/'runtime/future_repose_overlay.json'
 if path.exists() and json.loads(path.read_text()).get('schema_version')==2:
  verify_overlay(root)
  if config is None: config=json.loads((Path(root)/'configuracao_otimizacao.txt').read_text(encoding='utf-8-sig'))
  if config.get('fresh_finite_independent') is True:
   from .fresh_authorization import require_fresh_authorization
   require_fresh_authorization(root,config)
   return
  approval_path=Path(root)/'launch_authorization.json'
  approval=json.loads(approval_path.read_text()) if approval_path.exists() else {}
  if approval.get('scope') not in ('authorized-sequential-repose10-provenance-v1', 'authorized-sequential-dt100us-v1', 'fresh-doe84-dt100us-v1'):
   raise RuntimeError('historical repose angles are incompatible with ANSYS protocol; DOE blocked until fit/reuse isolation is implemented')
  from .launch_authorization import require_launch_authorization
  if config is None: config=json.loads((Path(root)/'configuracao_otimizacao.txt').read_text(encoding='utf-8-sig'))
  require_launch_authorization(root,config)

def verify_overlay(root):
 root=Path(root); meta=json.loads((root/'runtime/future_repose_overlay.json').read_text())
 timing=(meta.get('timestep_s'),meta.get('total_steps'))
 if meta.get('schema_version')!=2 or meta.get('protocol_revision')!='ansys-volumetric-once-total10s' or meta.get('preserve_prior_trials') is not True or meta.get('final_time_s')!=10.0 or timing not in ((5e-5,200000),(1e-4,100000)) or meta.get('reference_radius_m')!=.01 or meta.get('mass_multiplier') is not None: raise ValueError('invalid ANSYS future scope')
 if timing[0] == 1e-4:
  from optimizer.transition_guards import verify_candidate_pins
  verify_candidate_pins(root)
 # This is integrity validation only; it does not grant launch authorization.
 expected={'first_trial_id':556,'preserved_prepared_trials_through':555,'lift_start_s':1.,'lift_end_s':6.,'no_replenishment':True,'launch_authorized':False}
 if any(meta.get(k)!=v for k,v in expected.items()): raise ValueError('ANSYS overlay timing or historical scope mismatch')
 # Output-only authorization is separate from the original physical chronology.
 output_expected={'output_policy_revision':'repose-final-measurement-only-local-default-ignore-v1',
  'output_policy_first_trial_id':593,'output_policy_preserved_prepared_trials_through':592,
  'particle_dumps':'final-measurement-only','lost_particles_policy':'local-default-ignore'}
 if any(meta.get(k)!=v for k,v in output_expected.items()): raise ValueError('repose output policy scope mismatch')
 assets=dict(REVIEWED_ASSETS)
 if timing[0]==1e-4: assets['Ang Repouso/in.repouso']=CANDIDATE_REPOSE_SHA256
 if meta.get('revised_sha256')!=assets['Ang Repouso/in.repouso']: raise ValueError('repose output policy deck pin mismatch')
 # Code pins are immutable reviewed bytes, not self-rehashable sidecar authority.
 for rel,digest in assets.items():
  if sha(root/rel)!=digest or meta.get('asset_sha256',{}).get(rel)!=digest: raise ValueError('ANSYS overlay reviewed asset mismatch: '+rel)
 if sha(root/meta['baseline_relative_path'])!=meta['baseline_sha256']: raise ValueError('baseline hash mismatch')
 return True

from .transition_guards import CANDIDATE_SHA256
CANDIDATE_REPOSE_SHA256 = CANDIDATE_SHA256['Ang Repouso/in.repouso']
REVIEWED_ASSETS={'Ang Repouso/in.repouso': 'be73d9dcc3960501fd9a79aa6386d49b59f46dea9def65c6e0351b699154e122', 'Ang Repouso/initial_particles.data': '93972e909e05bd9e9b6d90e3699b63b6664a2b77b8aad0be5d19041229d76438', 'Ang Repouso/initial_audit.json': '888d5e2e8550e348ae38c08a774ca36c26f40c9341966c794d3435799ba054a6', 'Ang Repouso/meshes/Container_Cylinder.stl': 'ee5371a153e0db854b1029983e720e551b3c330f8436f4ec50092133404f1de4', 'Ang Repouso/meshes/Upper_Tray.stl': '6c11c3c7f53205e3ab276c6b500a40abf30a9378133bb703d6459adf473768d0'}
