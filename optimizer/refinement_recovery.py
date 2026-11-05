"""Safely continue the persisted DEM campaign after an interrupted host shutdown."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from optimizer.calibration_controller import CalibrationController

MANIFEST_NAME = "refinement_recovery_manifest.json"
EXPECTED_DOE_IDS = set(range(1, 46))
EXPECTED_COMPLETED_REFINEMENT_IDS = set(range(46, 70))
INTERRUPTED_IDS = set(range(70, 82))
FIRST_RECOVERY_ID = 94


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_parameters(left: dict, right: dict, names: tuple[str, ...]) -> bool:
    return all(math.isclose(float(left[name]), float(right[name]), rel_tol=0.0, abs_tol=1e-10) for name in names)


def pending_refinement_jobs(results: list[dict], jobs: list[dict], names: tuple[str, ...], phase: str = "refinement") -> list[dict]:
    """Exclude work already persisted as complete, so retries never duplicate valid results."""
    complete_keys = {
        tuple(round(float(row[name]), 12) for name in names)
        for row in results
        if row.get("phase") == phase and row.get("status") == "complete"
    }
    return [job for job in jobs if tuple(round(float(job["parameters"][name]), 12) for name in names) not in complete_keys]


def _active_campaign_processes(run_root: Path) -> list[dict]:
    """Find solver/controller processes tied to this run using /proc cwd and command line."""
    root = str(Path(run_root).resolve())
    active = []
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = str((entry / "cwd").resolve())
            command = (entry / "cmdline").read_bytes().decode(errors="replace").replace(chr(0), " ")
            comm = (entry / "comm").read_text(errors="replace").strip().lower()
            lower = command.lower()
            solver = comm in {"liggghts", "lmp_auto"} and cwd.startswith(root)
            runner = comm.startswith("python") and "optimizer.refinement_recovery" in lower and root in lower
            if solver or runner:
                active.append({"pid": int(entry.name), "cwd": cwd, "command": command[:240]})
        except (OSError, PermissionError):
            continue
    return active


def _next_unused_trial_id(run_root: Path, results: list[dict], statuses: list[dict]) -> int:
    ids = [int(row["trial_id"]) for row in results + statuses if row.get("trial_id", "").isdigit()]
    ids.extend(int(path.name[6:]) for path in Path(run_root).glob("trial_*") if path.name.startswith("trial_") and path.name[6:].isdigit())
    return max(ids, default=0) + 1


def _save_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    temporary.replace(path)


def build_recovery_plan(controller: CalibrationController, run_root: Path) -> dict:
    """Validate the persisted snapshot and construct a no-write plan for unused trial IDs."""
    if 'rolling' in [t[0] for t in controller.tests]:
        return _configured_plan(controller, Path(run_root))
    run_root = Path(run_root)
    results = _read_csv(run_root / "results.csv")
    statuses = _read_csv(run_root / "status.csv")
    latest = {row["trial_id"]: row for row in statuses}

    doe_rows = [row for row in results if row.get("phase") == "doe" and row.get("status") == "complete"]
    completed_refinement = [row for row in results if row.get("phase") == "refinement" and row.get("status") == "complete"]
    doe_ids = {int(row["trial_id"]) for row in doe_rows}
    refinement_ids = {int(row["trial_id"]) for row in completed_refinement}
    if doe_ids != EXPECTED_DOE_IDS:
        raise ValueError(f"expected complete DOE IDs 1–45; found {sorted(doe_ids)}")
    if refinement_ids != EXPECTED_COMPLETED_REFINEMENT_IDS:
        raise ValueError(f"expected complete refinement IDs 46–69; found {sorted(refinement_ids)}")
    if set(map(int, latest)) != set(range(1, 82)):
        raise ValueError("status.csv trial IDs do not match expected 1–81 snapshot")
    for trial_id in EXPECTED_DOE_IDS | EXPECTED_COMPLETED_REFINEMENT_IDS:
        if latest[str(trial_id)].get("status") != "complete":
            raise ValueError(f"trial {trial_id} is not latest-status complete")
    for trial_id in INTERRUPTED_IDS:
        row = latest[str(trial_id)]
        if row.get("phase") != "refinement" or row.get("status") != "running":
            raise ValueError(f"trial {trial_id} is not the expected interrupted refinement")
    if any(int(row["trial_id"]) in INTERRUPTED_IDS for row in results):
        raise ValueError("an interrupted refinement trial already has a results.csv row")

    candidates = controller.nominate_refinement_candidates(doe_rows)
    if len(candidates) != 48:
        raise ValueError(f"expected 48 original refinement candidates; found {len(candidates)}")
    for trial_id in range(46, 82):
        parameter_path = run_root / f"trial_{trial_id:04d}" / "parameters.json"
        if not parameter_path.is_file():
            raise FileNotFoundError(f"missing source parameters: {parameter_path}")
        saved = json.loads(parameter_path.read_text(encoding="utf-8"))["parameters"]
        if not _same_parameters(saved, candidates[trial_id - 46], controller.parameter_names):
            raise ValueError(f"reconstructed candidate differs from trial_{trial_id:04d}")

    validation_count = int(controller.config["campaign"]["validation_trials"])
    planned_ids = list(range(FIRST_RECOVERY_ID, FIRST_RECOVERY_ID + 24 + validation_count))
    collisions = [trial_id for trial_id in planned_ids if (run_root / f"trial_{trial_id:04d}").exists()]
    if collisions:
        raise FileExistsError(f"new trial directories already exist: {collisions}")
    recorded_ids = {int(row["trial_id"]) for row in results + statuses}
    if any(trial_id in recorded_ids for trial_id in planned_ids):
        raise FileExistsError("planned new trial IDs already appear in CSVs")

    jobs = []
    for offset, source_id in enumerate(range(70, 82)):
        candidate_index = source_id - 46
        trial_id = FIRST_RECOVERY_ID + offset
        jobs.append({"trial_id": trial_id, "source_trial_id": source_id, "candidate_index": candidate_index, "parameters": candidates[candidate_index], "seeds": controller._trial_seeds(trial_id)})
    for offset, candidate_index in enumerate(range(36, 48), start=12):
        trial_id = FIRST_RECOVERY_ID + offset
        jobs.append({"trial_id": trial_id, "source_trial_id": candidate_index + 46, "candidate_index": candidate_index, "parameters": candidates[candidate_index], "seeds": controller._trial_seeds(trial_id)})

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
        "baseline": {name: {"sha256": _sha256(run_root / name), "size_bytes": (run_root / name).stat().st_size} for name in ("status.csv", "results.csv")},
        "preserved_complete_counts": {"doe": len(doe_rows), "refinement": len(completed_refinement)},
        "refinement_jobs": jobs,
        "validation_count": validation_count,
        "validation_trial_ids": list(range(FIRST_RECOVERY_ID + 24, FIRST_RECOVERY_ID + 24 + validation_count)),
        "state": "planned",
    }


def _write_manifest_exclusive(path: Path, payload: dict) -> None:
    """Create the manifest once and refuse to overwrite a prior recovery record."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()


def execute_recovery(controller: CalibrationController, run_root: Path, plan: dict) -> dict:
    if not controller.config.get('simulation_launch_enabled', False):
        raise RuntimeError('simulation launching disabled; recovery is offline only')
    if plan.get('schema_version') == 2:
        _write_manifest_exclusive(Path(run_root)/MANIFEST_NAME,plan)
        return _configured_resume(controller,Path(run_root),plan)
    run_root = Path(run_root)
    manifest_path = run_root / MANIFEST_NAME
    active = _active_campaign_processes(run_root)
    if active:
        raise RuntimeError(f"campaign process already active; refusing duplicate launch: {active}")
    _write_manifest_exclusive(manifest_path, plan)
    backup = run_root / "pre_recovery_backup"
    backup.mkdir(exist_ok=False)
    for name in ("status.csv", "results.csv", "status_execucao.csv"):
        source = run_root / name
        if source.exists():
            shutil.copy2(source, backup / name)

    jobs = plan["refinement_jobs"]
    controller._run_points([job["parameters"] for job in jobs], "refinement", FIRST_RECOVERY_ID)
    scored = controller.score_completed_results()
    completed_refinement = [row for row in scored if row.get("phase") == "refinement" and row.get("status") == "complete"]
    if len(completed_refinement) < 48:
        plan["state"] = "refinement_incomplete"
        plan["refinement_complete"] = len(completed_refinement)
        manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"refinement has only {len(completed_refinement)}/48 complete trials; validation not started")

    best = min((row for row in scored if row.get("status") == "complete"), key=lambda row: (float(row["score"]), int(row["trial_id"])))
    validation_parameters = [{name: float(best[name]) for name in controller.parameter_names} for _ in range(plan["validation_count"])]
    validation_first_id = controller._next_trial_id
    plan["validation_first_trial_id"] = validation_first_id
    plan["state"] = "validation_running"
    manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    controller._run_points(validation_parameters, "validation", validation_first_id)
    final_rows = controller.score_completed_results()
    validation_complete = sum(row.get("phase") == "validation" and row.get("status") == "complete" for row in final_rows)
    plan.update({"state": "complete" if validation_complete == plan["validation_count"] else "validation_incomplete", "validation_complete": validation_complete, "final_results_rows": len(final_rows)})
    manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"state": plan["state"], "manifest": str(manifest_path), "refinement_complete": len(completed_refinement), "validation_complete": validation_complete, "results_rows": len(final_rows)}


def resume_recovery(controller: CalibrationController, run_root: Path) -> dict:
    """Continue only missing parameter points, assigning fresh IDs after interrupted attempts."""
    run_root = Path(run_root).resolve()
    manifest_path = run_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no recovery manifest at {manifest_path}")
    active = _active_campaign_processes(run_root)
    if active:
        raise RuntimeError(f"campaign process still active; refusing duplicate launch: {active}")
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    if plan.get('schema_version') == 2:
        if not controller.config.get('simulation_launch_enabled', False):
            raise RuntimeError('simulation launching disabled; recovery is offline only')
        return _configured_resume(controller,run_root,plan)
    if Path(plan["run_root"]).resolve() != run_root:
        raise ValueError("recovery manifest points to a different run root")
    if plan.get("state") == "complete":
        return {"state": "complete", "manifest": str(manifest_path), "message": "campaign already complete"}
    results = _read_csv(run_root / "results.csv")
    statuses = _read_csv(run_root / "status.csv")
    jobs = plan["refinement_jobs"]
    pending = pending_refinement_jobs(results, jobs, controller.parameter_names)
    if pending:
        first_id = _next_unused_trial_id(run_root, results, statuses)
        plan.setdefault("resume_attempts", []).append({"phase": "refinement", "first_trial_id": first_id, "candidate_indices": [job["candidate_index"] for job in pending]})
        plan["state"] = "refinement_running"
        _save_manifest(manifest_path, plan)
        controller._run_points([job["parameters"] for job in pending], "refinement", first_id)
        results = controller.score_completed_results()
        pending = pending_refinement_jobs(results, jobs, controller.parameter_names)
    if pending:
        plan["state"] = "refinement_incomplete"
        plan["refinement_complete"] = len(jobs) - len(pending)
        _save_manifest(manifest_path, plan)
        return {"state": plan["state"], "refinement_complete": plan["refinement_complete"], "refinement_total": len(jobs), "manifest": str(manifest_path)}
    plan["refinement_complete"] = len(jobs)
    scored = controller.score_completed_results()
    validation_parameters = plan.get("validation_parameters")
    if validation_parameters is None:
        best = min((row for row in scored if row.get("status") == "complete"), key=lambda row: (float(row["score"]), int(row["trial_id"])))
        validation_parameters = {name: float(best[name]) for name in controller.parameter_names}
        plan["validation_parameters"] = validation_parameters
    validation_count = int(plan["validation_count"])
    validation_complete = sum(row.get("phase") == "validation" and row.get("status") == "complete" and _same_parameters(row, validation_parameters, controller.parameter_names) for row in scored)
    remaining = max(0, validation_count - validation_complete)
    if remaining:
        current_statuses = _read_csv(run_root / "status.csv")
        first_id = _next_unused_trial_id(run_root, scored, current_statuses)
        plan.setdefault("resume_attempts", []).append({"phase": "validation", "first_trial_id": first_id, "count": remaining})
        plan["state"] = "validation_running"
        _save_manifest(manifest_path, plan)
        controller._run_points([validation_parameters for _ in range(remaining)], "validation", first_id)
        scored = controller.score_completed_results()
        validation_complete = sum(row.get("phase") == "validation" and row.get("status") == "complete" and _same_parameters(row, validation_parameters, controller.parameter_names) for row in scored)
    plan.update({"state": "complete" if validation_complete == validation_count else "validation_incomplete", "validation_complete": validation_complete, "final_results_rows": len(scored)})
    _save_manifest(manifest_path, plan)
    return {"state": plan["state"], "refinement_complete": len(jobs), "validation_complete": validation_complete, "results_rows": len(scored), "manifest": str(manifest_path)}


def _configuration_identity(controller):
    keys=('tests','parameters','weights','response_scales','campaign','maximum_doe_attempts')
    values={k:controller.config.get(k) for k in keys}
    values.update({k:v for k,v in controller.config.items() if k.startswith('target_')})
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def _configured_plan(controller, root):
    """New campaigns use configured DOE size, finite responses and parameter keys."""
    rows=_read_csv(root/'results.csv') if (root/'results.csv').exists() else []
    doe=[r for r in rows if r.get('phase')=='doe' and controller._valid_responses(r)]
    minimum=int(controller.config.get('minimum_doe_complete',controller.config['campaign']['doe_trials']))
    if not controller._doe_is_ready(rows):
        raise ValueError(f'expected at least {minimum} finite complete DOE trials, including rolling')
    # Recompute fixed-scale scores in memory; a dry-run must not rewrite CSVs.
    from optimizer.article_optimizer import weighted_normalized_score
    for r in rows:
        if controller._valid_responses(r):
            errors={k:abs(float(r[k+'_deg'])-controller.config['target_'+k+'_deg']) for k,*_ in controller.tests}
            r['score']=weighted_normalized_score(errors,controller.response_scales,controller.config['weights'])
    points=controller.nominate_refinement_candidates(doe)
    jobs=[dict(candidate_index=i,parameters=p) for i,p in enumerate(points)]
    valid=[r for r in rows if controller._valid_responses(r)]
    pending=pending_refinement_jobs(valid,jobs,controller.parameter_names)
    return dict(schema_version=2,run_root=str(root.resolve()),configuration_identity=_configuration_identity(controller),
                state='planned',refinement_jobs=jobs,remaining_refinement=len(pending),
                preserved_complete_counts=dict(doe=len(doe),refinement=len(jobs)-len(pending)),
                validation_count=min(5,int(controller.config['campaign']['validation_trials'])),
                physical_validation_performed=False)


def _configured_resume(controller, root, plan):
    if Path(plan['run_root']).resolve()!=root.resolve() or plan['configuration_identity']!=_configuration_identity(controller):
        raise ValueError('recovery configuration or run root differs from saved plan')
    path=root/MANIFEST_NAME
    if plan.get('state')=='complete': return dict(state='complete',manifest=str(path))
    if _active_campaign_processes(root): raise RuntimeError('campaign process already active; refusing duplicate launch')
    rows=controller.score_completed_results()
    pending=pending_refinement_jobs([r for r in rows if controller._valid_responses(r)],plan['refinement_jobs'],controller.parameter_names)
    if pending:
        statuses=_read_csv(root/'status.csv') if (root/'status.csv').exists() else []
        first=_next_unused_trial_id(root,rows,statuses)
        plan['state']='refinement_running';_save_manifest(path,plan)
        controller._run_points([j['parameters'] for j in pending],'refinement',first)
        rows=controller.score_completed_results()
        pending=pending_refinement_jobs([r for r in rows if controller._valid_responses(r)],plan['refinement_jobs'],controller.parameter_names)
    if pending:
        plan.update(state='refinement_incomplete',remaining_refinement=len(pending));_save_manifest(path,plan)
        return dict(state=plan['state'],remaining_refinement=len(pending),manifest=str(path))
    if 'validation_candidates' not in plan:
        candidates=controller.promising_candidates(rows)
        (root/'promising_candidates.json').write_text(json.dumps(dict(candidates=candidates,
            physical_validation_performed=False),indent=2),encoding='utf8')
        plan['validation_candidates']=[{n:float(r[n]) for n in controller.parameter_names}
            for r in candidates[:plan['validation_count']]]
        _save_manifest(path,plan)
    jobs=[dict(parameters=p) for p in plan['validation_candidates']]
    pending=pending_refinement_jobs([r for r in rows if controller._valid_responses(r)],jobs,controller.parameter_names,phase='validation')
    if pending:
        statuses=_read_csv(root/'status.csv') if (root/'status.csv').exists() else []
        first=_next_unused_trial_id(root,rows,statuses)
        plan['state']='validation_running';_save_manifest(path,plan)
        controller._run_points([j['parameters'] for j in pending],'validation',first)
        rows=controller.score_completed_results()
    pending=pending_refinement_jobs([r for r in rows if controller._valid_responses(r)],jobs,controller.parameter_names,phase='validation')
    plan.update(state='validation_incomplete' if pending else 'complete',validation_complete=len(jobs)-len(pending))
    controller.write_validation_summary(rows,len(jobs))
    _save_manifest(path,plan)
    return dict(state=plan['state'],validation_complete=plan['validation_complete'],manifest=str(path),physical_validation_performed=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    modes = (args.dry_run, args.launch, args.resume)
    if sum(modes) != 1:
        parser.error("choose exactly one of --dry-run, --launch, or --resume")
    controller = CalibrationController(args.project_root, args.run_root)
    print(json.dumps(controller.preflight(), indent=2))
    if args.resume:
        print(json.dumps(resume_recovery(controller, args.run_root), indent=2))
        return 0
    plan = build_recovery_plan(controller, args.run_root)
    if args.dry_run:
        report = {"state": "dry_run_ok", "preserved_complete_counts": plan["preserved_complete_counts"],
                  "validation_count": plan["validation_count"], "manifest_path": str(Path(args.run_root) / MANIFEST_NAME)}
        if plan.get('schema_version') == 2:
            report.update(candidate_indices=[job['candidate_index'] for job in plan['refinement_jobs']],
                          remaining_refinement=plan['remaining_refinement'])
        else:
            report['recovery_ids'] = [job['trial_id'] for job in plan['refinement_jobs']]
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(execute_recovery(controller, args.run_root, plan), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
