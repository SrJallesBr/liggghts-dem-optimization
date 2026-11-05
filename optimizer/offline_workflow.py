"""Reproducible plan, prepared decks or offline tests; no solver entry point."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from optimizer.article_optimizer import latin_hypercube
from optimizer.calibration_controller import CalibrationController


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('offline_demo'))
    parser.add_argument('--parallel-trials',type=int,default=1)
    parser.add_argument('action',choices=['plan','prepare','verify'])
    parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--trial-id',type=int,default=1)
    args=parser.parse_args(argv)
    root=args.project_root.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.action=='verify':
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        temporary=output/('test_artifacts_'+stamp)
        command=[sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider','--basetemp='+str(temporary)]
        env=dict(os.environ,MPLCONFIGDIR=str(output/'matplotlib_cache'))
        proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
        report=dict(command=command,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
                    offline_only=True,real_dem_started=False)
        (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(report,indent=2))
        return proc.returncode
    controller=CalibrationController(root,output/'prepared',parallel_trials_override=args.parallel_trials)
    preflight=controller.preflight()
    if args.action=='plan':
        bounds={k:(s['min'],s['max']) for k,s in controller.config['parameters'].items()}
        points=latin_hypercube(args.count,bounds,controller.config['random_seed'])
        report=dict(points=points,campaign=controller.config['campaign'],preflight=preflight,
                    configured_tests=[t[0] for t in controller.tests],
                    offline_only=True,real_dem_started=False,
                    note='Small preparation demonstration only; no synthetic or physical response angles assigned.')
        (output/'offline_plan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    else:
        plan=json.loads((output/'offline_plan.json').read_text(encoding='utf8'))
        if not 1 <= args.trial_id <= len(plan['points']):
            parser.error('trial-id must identify a point in offline_plan.json')
        workspace=controller.prepare_trial(args.trial_id,plan['points'][args.trial_id-1])
        report=dict(workspace=str(workspace),seeds=controller._trial_seeds(args.trial_id),
                    offline_only=True,real_dem_started=False)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
