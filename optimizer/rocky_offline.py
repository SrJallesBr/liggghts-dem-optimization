"""Offline batch remeasurement of configured saved trials. No solver option."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from optimizer.calibration_controller import CalibrationController
from optimizer.rocky_io import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inventory',type=Path)
    group.add_argument('--trial',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(argv)
    entries=json.loads(args.inventory.read_text(encoding='utf8')) if args.inventory else [{'trial':str(args.trial)}]
    sources=list(dict.fromkeys(str(Path(e['trial']).resolve()) for e in entries))
    output=args.output.resolve()
    for source in sources:
        if output.is_relative_to(Path(source)):
            parser.error('output must not be inside any archived source trial')
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be new or empty; previous reports are preserved')
    output.mkdir(parents=True,exist_ok=True)
    controller=CalibrationController(args.project_root,output/'controller_state')
    # Hash input deck/mesh/snapshots before and after; differences are errors.
    source_files={str(p.resolve()) for source in sources for p in Path(source).rglob('*')
                  if p.is_file() and (p.suffix.lower() in {'.vtk','.stl','.dump'} or p.name.startswith('in.'))}
    before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(source_files)}
    rows=[]
    for i,source in enumerate(sources,1):
        trial=Path(source); folder=output/f'{i:03d}__{trial.parent.name}__{trial.name}'
        result=controller.reprocess_pair(trial,folder)
        rows.append(result)
        print(json.dumps({'index':i,'source':source,'status':result['status'],
                          'angles':{k:v['angle_deg'] for k,v in result['responses'].items()}},ensure_ascii=False),flush=True)
    after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(source_files)}
    changed=[p for p in before if before[p]!=after[p]]
    kinds=[t[0] for t in controller.tests]
    counts={k:dict(Counter(r['responses'][k]['status'] for r in rows)) for k in kinds}
    fingerprints=[tuple(r['responses'][k].get('source_sha256') for k in kinds) for r in rows]
    summary={'pairs_processed':len(rows),'pairs_measured':sum(r['status']=='measured' for r in rows),
             'unique_snapshot_pairs':len({v for v in fingerprints if all(v)}),
             'status_counts':counts,'configured_tests':kinds,'real_dem_started':False,'offline_only':True,
             'method':controller.config['measurement'],'originals_unchanged':not changed,
             'source_files_hashed':len(before),'changed_source_files':changed,'pairs':rows}
    write_json(output/'summary.json',summary)
    write_json(output/'source_hashes.json',{'before':before,'after':after,'changed':changed})
    with (output/'summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['source_trial','pair_status']+[k+s for k in kinds for s in ['_status','_deg']])
        for r in rows:
            a=r['responses'];w.writerow([r['source_workspace'],r['status']]+[a[k][v] for k in kinds for v in ['status','angle_deg']])
    from optimizer.rocky_audit import write_gallery
    write_gallery(output)
    print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},ensure_ascii=False,indent=2))
    return 2 if changed else 0


if __name__=='__main__':
    raise SystemExit(main())
