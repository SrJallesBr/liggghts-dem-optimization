"""Read-only campaign watcher that creates measurement SVGs as trial VTKs settle."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from optimizer.calibration_controller import CalibrationController


ARTIFACTS = {
    "discharge": ("Ang Deslizamento/post/deslizamento_final.vtk", "medicao_descarga.svg"),
    "repose": ("Ang Repouso/post/repouso_10s.vtk", "medicao_repouso_oito_direcoes.svg"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--max-hours", type=float, default=36)
    args = parser.parse_args()
    controller = CalibrationController(args.project_root, run_root=args.run_root, solver_command=["true"])
    trial_dirs = sorted(args.run_root.glob("trial_*"))
    stability: dict[Path, tuple[int, int]] = {}
    deadline = time.monotonic() + args.max_hours * 3600
    print(f"watching {len(trial_dirs)} trial folders under {args.run_root}", flush=True)
    while time.monotonic() < deadline:
        pending = 0
        for trial in trial_dirs:
            for kind, (relative, output_name) in ARTIFACTS.items():
                out = trial / output_name
                diagnostic = trial / f"{output_name}.diagnostic.txt"
                if (out.is_file() and out.stat().st_size > 0) or diagnostic.is_file():
                    continue
                source = trial / relative
                if not source.is_file() or source.stat().st_size == 0:
                    pending += 1
                    continue
                size = source.stat().st_size
                previous_size, stable_count = stability.get(source, (-1, 0))
                stable_count = stable_count + 1 if size == previous_size else 0
                stability[source] = (size, stable_count)
                if stable_count < args.stable_checks:
                    pending += 1
                    continue
                try:
                    result = controller._measure_snapshot(source, kind)
                    print(f"{trial.name}: {kind} visual written ({result['mean_angle_deg']:.3f} deg)", flush=True)
                except Exception as exc:
                    diagnostic = trial / f"{output_name}.diagnostic.txt"
                    diagnostic.write_text(f"Could not render {kind} measurement from {source}: {type(exc).__name__}: {exc}\n", encoding="utf-8")
                    print(f"{trial.name}: {kind} unavailable: {type(exc).__name__}: {exc}", flush=True)
        if pending == 0:
            print("all expected measurement artifacts have been rendered or diagnosed", flush=True)
            return 0
        time.sleep(max(2, args.poll_seconds))
    print("watch deadline reached; any unrendered artifacts remain pending", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
