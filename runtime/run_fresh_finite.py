#!/usr/bin/env python3
"""Fresh independent finite campaign: DOE, backfill and bounded refinement.
No history migration/recovery from another campaign is permitted.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
sys.path.insert(0,str(ROOT))
from optimizer.calibration_controller import CalibrationController
from optimizer.adaptive_refinement import run_adaptive_refinement
from optimizer.fresh_authorization import require_fresh_authorization,SCOPE


def verify_isolation(root):
    root=Path(root).resolve();run=root/'run'
    if run.is_symlink():raise RuntimeError('HOLD: run directory must not be a symlink')
    if (run/'import_manifest.json').exists():raise RuntimeError('HOLD: historical import forbidden')
    config=json.loads((root/'configuracao_otimizacao.txt').read_text())
    first=int(config['first_trial_id']);owners={}
    for filename,phase in [('doe_schedule.jsonl','doe'),('adaptive_proposals.jsonl','refinement')]:
        journal=run/filename
        if not journal.exists():continue
        for line in journal.read_text().splitlines():
            e=json.loads(line)
            if e.get('event')!='reserved':continue
            tid=int(e['trial_id'])
            if tid in owners or not first<=tid<first+1000:raise RuntimeError('HOLD: duplicate or out-of-budget reservation')
            if phase=='doe' and tid>=first+420:raise RuntimeError('HOLD: DOE attempt cap exceeded')
            owners[tid]=phase
    for filename in ['results.csv','scores.csv','trials.csv','status.csv']:
        p=run/filename
        if p.exists():
            with p.open(newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('trial_id'):
                        tid=int(row['trial_id'])
                        if tid not in owners or row.get('phase')!=owners[tid]:raise RuntimeError('HOLD: unowned historical row: '+filename)
                        if row.get('workspace') and not Path(row['workspace']).resolve().is_relative_to(run.resolve()):raise RuntimeError('HOLD: external historical workspace')
    for p in run.glob('trial_*'):
        if p.is_symlink() or not p.name[6:].isdigit() or int(p.name[6:]) not in owners:raise RuntimeError('HOLD: unowned trial workspace: '+str(p))
    return owners


def run(c):
    from optimizer.adaptive_refinement import _configured_budget_limit
    refinement=[]
    def handoff(live):
        budget=_configured_budget_limit(c,doe_records=live['records'])
        refinement.append(run_adaptive_refinement(c,workers=int(c.config['parallel_trials']),budget=budget,doe_handoff=live))
    rows=c.run_doe(handoff=handoff)
    if refinement:
        summary=refinement[0]
        return {'status':summary['status'],'refinement_started':True,'refinement':summary}
    if not c._doe_is_ready(rows):return {'status':'doe_insufficient','refinement_started':False}
    budget=_configured_budget_limit(c)
    summary=run_adaptive_refinement(c,workers=int(c.config['parallel_trials']),budget=budget)
    return {'status':summary['status'],'refinement_started':True,'refinement':summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Check only: never launches solver')
    mode.add_argument('--start',action='store_true',help='Explicitly start/resume this independent campaign')
    parser.add_argument('--solver',help='Path to existing LIGGGHTS executable (default: /usr/local/bin/liggghts)')
    args=parser.parse_args()
    if os.name!='posix':raise RuntimeError('Run inside Linux/WSL, not Windows Python')
    # Set limits before numpy/scipy are imported by model fitting in workers.
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    import fcntl
    with (ROOT/'.fresh-doe.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p in Path('/proc').iterdir():
            if p.name.isdigit() and int(p.name)!=os.getpid():
                try:cwd=(p/'cwd').resolve(strict=True)
                except OSError:continue
                if cwd.is_relative_to(ROOT/'run'):raise RuntimeError('HOLD: process in run workspace; reconcile before restart: '+p.name)
        verify_isolation(ROOT)
        c=CalibrationController(ROOT,run_root=ROOT/'run')
        os.environ['DEM_FRESH_FINITE_AUTHORIZATION_ID']=SCOPE
        require_fresh_authorization(ROOT,c.config)
        report=c.preflight()
        solver=shutil.which(args.solver or c.config['solver'])
        if not solver:raise RuntimeError('HOLD: LIGGGHTS executable not found; use --solver /existing/path')
        c.solver_command=[solver]  # Exact argument vector, never a shell string.
        report['solver']=solver
        print(json.dumps(report,indent=2),flush=True)
        if args.start:
            result=run(c)
            (ROOT/'run/finite_summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2),flush=True)
            if result['status'] in ('doe_insufficient','error','interrupted'):raise SystemExit(2)

if __name__=='__main__':main()
