"""Configured independent tests; local measurements; simulation disabled by default."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock

from optimizer.article_optimizer import latin_hypercube, recommend_campaign_size, weighted_normalized_score
from optimizer.cohesion_guard import CohesionLimitTracker
from optimizer.measurements import UnmeasurableAngle, measure_repose_eight_ray_least_squares, measure_v_sliding_angle, measure_v_sliding_angle_by_height, measure_v_sliding_angle_three_cuts, parse_ascii_stl_bounds, parse_vtk_points, parse_vtk_scalar_field
from optimizer.measurement_visuals import write_discharge_failure_svg, write_discharge_svg, write_measurement_diagnostic_svg, write_repose_failure_diagnostic_svg, write_repose_svg, write_svg_png

TESTS = (
    ("discharge", "Ang Deslizamento", "in.deslizamento", "post/deslizamento_final.vtk"),
    ("repose", "Ang Repouso", "in.repouso", "post/repouso_10s.vtk"),
    ("rolling", "Ang Rolamento", "in.rolamento", "post"),
)
PARAMETER_LINE = r"^(\s*variable\s+{name}\s+index\s+)(\S+)(.*)$"
SEED_PATTERNS = {
    "distribution": re.compile(r"(particledistribution/discrete\s+)(\d+)", re.IGNORECASE),
    "insertion": re.compile(r"(insert/stream\s+seed\s+)(\d+)", re.IGNORECASE),
}
STATUS_FIELDS = ("trial_id", "phase", "status", "stage", "message", "workspace", "started_at", "finished_at")
RESULT_FIELDS = ('trial_id', 'phase', 'status', 'sf11', 'rf11', 'rf12', 'CED11', 'distribution_seed', 'insertion_seed', 'measurement_method', 'discharge_response_convention', 'equivalence_to_ansys_verified', 'discharge_deg', 'discharge_left_deg', 'discharge_right_deg', 'discharge_paired_mean_deg', 'discharge_asymmetry_deg', 'repose_deg', 'discharge_error', 'repose_error', 'score', 'workspace')
RESULT_FIELDS += ('response_policy',) + tuple(
    kind + suffix for kind in ('discharge', 'repose', 'rolling')
    for suffix in ('_raw_deg', '_raw_status', '_raw_reason', '_response_status',
                   '_measurement_valid', '_assignment_reason'))
RESULT_FIELDS += ('sf12', 'CED12', 'rolling_deg', 'rolling_error', 'accepted', 'rolling_qc_json')
RESULT_FIELDS += tuple(kind+suffix for kind in ('discharge','repose','rolling')
                       for suffix in ('_physical_outcome','_physical_outcome_verified','_physical_outcome_evidence'))
DIAGNOSTIC_FIELDS = ("trial_id", "phase", "status", "reason", "parameters_json", "workspace")
LIVE_STATUS_FIELDS = ("trial", "phase", "status", "simulation_time_s", "started_at", "elapsed_seconds", "updated_at")


def simulation_time_seconds(input_file: Path, log_file: Path) -> float:
    """Calcula o último tempo físico a partir de Step × dt no log do LIGGGHTS."""
    try:
        text = input_file.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*variable\s+dt\s+equal\s+([^\s#]+)", text)
        if not match:
            return 0.0
        timestep = float(match.group(1))
        latest_step = None
        thermo = False
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Step" in line and "Atoms" in line:
                thermo = True
                continue
            if thermo:
                step = re.match(r"^\s*(\d+)\s+", line)
                if step:
                    latest_step = int(step.group(1))
        return round((latest_step or 0) * timestep, 1)
    except (OSError, ValueError):
        return 0.0


def refresh_live_state(run_root: Path, active: dict[int, dict]) -> None:
    """Atualiza estágio e tempo físico a partir dos logs produzidos por cada trial ativo."""
    for trial_id, state in active.items():
        workspace = run_root / f"trial_{trial_id:04d}"
        v_input = workspace / "Ang Deslizamento" / "in.deslizamento"
        repose_input = workspace / "Ang Repouso" / "in.repouso"
        v_log = workspace / "discharge.log"
        repose_log = workspace / "repose.log"
        rolling_log = workspace / 'rolling.log'
        fill_log = workspace / 'rolling_fill.log'
        if rolling_log.is_file():
            state.update(status='rolling', simulation_time_s=simulation_time_seconds(workspace/'Ang Rolamento/in.rolamento', rolling_log))
        elif fill_log.is_file():
            state.update(status='rolling_fill', simulation_time_s=simulation_time_seconds(workspace/'Ang Rolamento/in.fill_1s', fill_log))
        elif repose_log.is_file():
            state["status"] = "repose"
            state["simulation_time_s"] = simulation_time_seconds(repose_input, repose_log)
        elif v_log.is_file():
            state["status"] = "discharge"
            state["simulation_time_s"] = simulation_time_seconds(v_input, v_log)


def write_live_status_csv(root: Path, active: dict[int, dict], now: float | None = None) -> None:
    """Grava status_execucao.csv atomicamente, com tentativas para pastas sincronizadas."""
    now = time.time() if now is None else now
    target = root / "status_execucao.csv"
    temporary = root / ".status_execucao.csv.tmp"
    for attempt in range(3):
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LIVE_STATUS_FIELDS)
                writer.writeheader()
                for trial_id, state in sorted(active.items()):
                    started = state["started_at"]
                    writer.writerow({"trial": f"trial_{trial_id:04d}", "phase": state["phase"], "status": state["status"], "simulation_time_s": f"{state.get('simulation_time_s', 0.0):.1f}", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)), "elapsed_seconds": str(int(now - started)), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))})
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 2:
                return
            time.sleep(0.25 * (attempt + 1))


def render_dashboard(active: dict[int, dict], completed: int, maximum: int, now: float | None = None, interval: int = 10) -> None:
    """Exibe o painel ao vivo no terminal, no mesmo formato do otimizador anterior."""
    now = time.time() if now is None else now
    print("=== PAINEL DA OTIMIZACAO DEM ===")
    print(f"Concluidos: {completed}/{maximum} | Atualiza a cada {interval}s | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
    print(f"{'TRIAL':<18} {'FASE':<22} {'STATUS':<18} {'SIM_TIME':<10} TEMPO REAL")
    print("-" * 82)
    for trial_id, state in sorted(active.items()):
        sim_time = f"{state.get('simulation_time_s', 0.0):.1f}".replace(".", ",")
        elapsed = int(now - state["started_at"])
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{('trial_' + f'{trial_id:04d}'):<18} {state['phase']:<22} {state['status']:<18} {sim_time + 's':<10} {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("\nCSV ao vivo: status_execucao.csv | Resultados finalizados: results.csv", flush=True)


def is_valid_liggghts_seed(value: int) -> bool:
    """LIGGGHTS exige sementes primas maiores que 10000; manter abaixo de 900 milhões é compatível com o gerador interno."""
    value = int(value)
    if value <= 10_000 or value >= 900_000_000:
        return False
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def next_liggghts_prime(candidate: int) -> int:
    """Retorna a primeira semente prima válida no intervalo aceito pelo LIGGGHTS."""
    candidate = max(10_001, int(candidate) | 1)
    while candidate < 900_000_000:
        if is_valid_liggghts_seed(candidate):
            return candidate
        candidate += 2
    raise RuntimeError("could not allocate a valid LIGGGHTS random seed")


def load_configuration(path: Path) -> dict:
    """Lê a configuração raiz e deriva o tamanho da campanha pelos parâmetros."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config.get("parameters"), dict) or not config["parameters"]:
        raise ValueError("configuracao_otimizacao.txt requires non-empty parameters")
    defaults = recommend_campaign_size(config["parameters"])
    explicit_campaign = config.get("campaign", {})
    if explicit_campaign and not isinstance(explicit_campaign, dict):
        raise ValueError("campaign must be an object when provided")
    legacy_campaign = {name: config[name] for name in ("doe_trials", "refinement_trials", "validation_trials", "max_trials") if name in config}
    config["campaign"] = {**defaults, **legacy_campaign, **explicit_campaign}
    for name in legacy_campaign.keys() & explicit_campaign.keys():
        if legacy_campaign[name] != explicit_campaign[name]:
            raise ValueError(f'root/campaign mismatch: {name}')
    required_total = sum(int(config["campaign"][name]) for name in ("doe_trials", "refinement_trials", "validation_trials"))
    if 'doe_attempt_multiplier' in config:
        multiplier = config['doe_attempt_multiplier']
        if type(multiplier) is not int or multiplier not in (1, 5):
            raise ValueError('DOE attempt multiplier must be integer 1 or 5')
        cap = config.get('maximum_doe_attempts')
        if type(cap) is not int or cap != multiplier * int(config['campaign']['doe_trials']):
            raise ValueError('maximum_doe_attempts must equal the DOE target times the multiplier')
        if multiplier == 5:
            if (int(config['campaign']['doe_trials']) != 84
                    or int(config['campaign']['refinement_trials']) != 748
                    or int(config['campaign']['validation_trials']) != 0
                    or int(config['campaign']['max_trials']) != 1000):
                raise ValueError('invalid finite continuation campaign budget')
            required_total = 1000
        else:
            required_total = cap + int(config['campaign']['refinement_trials']) + int(config['campaign']['validation_trials'])
    configured_sizes = {name for name in ("doe_trials", "refinement_trials", "validation_trials", "max_trials") if name in legacy_campaign or name in explicit_campaign}
    if {"doe_trials", "refinement_trials", "validation_trials", "max_trials"}.issubset(configured_sizes) and int(config["campaign"]["max_trials"]) != required_total:
        raise ValueError("campaign.max_trials must equal doe_trials + refinement_trials + validation_trials")
    tests = config.get('tests', ['discharge','repose'])
    if not isinstance(tests,list) or not tests or len(set(tests)) != len(tests) or set(tests)-{t[0] for t in TESTS}:
        raise ValueError('unknown or duplicate configured tests')
    scales = {k:config.get('response_scales',{}).get(k,config.get('tolerance_deg',.5)) for k in tests}
    weights = config.get('weights',{})
    weighted_normalized_score(dict.fromkeys(tests,0.),scales,weights)
    for kind in tests:
        if not math.isfinite(float(config['target_'+kind+'_deg'])):
            raise ValueError('finite response targets required')
    for spec in config['parameters'].values():
        if not all(math.isfinite(float(spec[k])) for k in ('min','max','initial')) or not spec['min'] < spec['max'] or not spec['min'] <= spec['initial'] <= spec['max']:
            raise ValueError('invalid factor bounds/initial')
    if 'rolling' in tests:
        from optimizer.rolling_protocol import normalize_protocol
        config['rolling_protocol'] = normalize_protocol(config.get('rolling_protocol', {'version':'V5'}))
        if int(config.get('mpi_processes',1)) != 1 or int(config.get('mpi_processes_per_trial',1)) != 1:
            raise ValueError('rolling requires one solver rank')
        if int(config['campaign']['validation_trials']) > 5:
            raise ValueError('at most five candidate/seed validation trials')
    if int(config.get('maximum_doe_attempts',config['campaign']['doe_trials'])) < int(config.get('minimum_doe_complete',config['campaign']['doe_trials'])):
        raise ValueError('DOE attempt cap below required complete DOE')
    return config


def rewrite_input_parameters(path: Path, parameters: dict[str, float]) -> None:
    """Substitui somente variáveis index explicitamente solicitadas no input copiado."""
    text = Path(path).read_text(encoding="utf-8")
    for name, value in parameters.items():
        pattern = re.compile(PARAMETER_LINE.format(name=re.escape(name)), re.MULTILINE)
        text, count = pattern.subn(lambda match: f"{match.group(1)}{value:.12g}{match.group(3)}", text, count=1)
        if count != 1:
            raise ValueError(f"variable {name!r} was not found exactly once in {path}")
    Path(path).write_text(text, encoding="utf-8")


def rewrite_input_seeds(path: Path, seeds: dict[str, int]) -> None:
    """Replaces the two stochastic seed sites in a copied input only."""
    text = Path(path).read_text(encoding="utf-8")
    for name, pattern in SEED_PATTERNS.items():
        if name not in seeds:
            raise ValueError(f"missing {name} seed")
        if not pattern.search(text):
            pattern = re.compile(rf'(?m)^(\s*variable\s+{name}_seed\s+index\s+)(\d+)')
        text, count = pattern.subn(lambda match: f"{match.group(1)}{int(seeds[name])}", text, count=1)
        if count != 1:
            raise ValueError(f"{name} seed site was not found exactly once in {path}")
    Path(path).write_text(text, encoding="utf-8")


def measure_pile_angle(points, bins=24, x_half_width=0.02, particle_radii=None) -> dict:
    """Measure four diametral repose cuts with eight base-anchored robust faces."""
    return measure_repose_eight_ray_least_squares(points, particle_radii=particle_radii)


def discharge_cut_result_fields(measurement: dict) -> dict:
    """Persist cut measurements by their physical coordinates, not section list order."""
    cut_names = {0.0: "x0", 0.1: "xplus01", -0.1: "xminus01"}
    sections_by_cut = {}
    for section in measurement.get("sections", []):
        if "section_center_physical" not in section:
            raise ValueError("discharge section is missing section_center_physical")
        center = float(section["section_center_physical"])
        matching = [value for value in cut_names if math.isclose(center, value, rel_tol=0.0, abs_tol=1e-9)]
        if len(matching) != 1:
            raise ValueError(f"unexpected discharge section center: {center}")
        cut_name = cut_names[matching[0]]
        if cut_name in sections_by_cut:
            raise ValueError(f"duplicate discharge section center: {center}")
        sections_by_cut[cut_name] = section
    if set(sections_by_cut) != set(cut_names.values()):
        raise ValueError("discharge results must contain the three physical cuts at -0.1, 0.0, and +0.1 m")
    fields = {}
    for cut_name, section in sections_by_cut.items():
        fields[f"discharge_{cut_name}_minimum_height_particle_diameters"] = float(section["minimum_height_particle_diameters"])
        fields[f"discharge_{cut_name}_left_profile_start_height_m"] = float(section["profile_start_height_left"])
        fields[f"discharge_{cut_name}_right_profile_start_height_m"] = float(section["profile_start_height_right"])
        fields[f"discharge_{cut_name}_deg"] = section["mean_angle_deg"]
        fields[f"discharge_{cut_name}_left_deg"] = section["left_angle_deg"]
        fields[f"discharge_{cut_name}_right_deg"] = section["right_angle_deg"]
        fields[f"discharge_{cut_name}_center_physical_m"] = float(section["section_center_physical"])
        fields[f"discharge_{cut_name}_band_clipped"] = section["band_clipped"]
        fields[f"discharge_{cut_name}_band_min_m"] = section["effective_band_physical"][0]
        fields[f"discharge_{cut_name}_band_max_m"] = section["effective_band_physical"][1]
        for side in ("left", "right"):
            segments = section[f"{side}_segments"]
            fields[f"discharge_{cut_name}_{side}_segment1_count"] = len(segments)
            for index in (1, 2):
                segment = segments[index - 1] if index <= len(segments) else None
                fields[f"discharge_{cut_name}_{side}_segment{index}_deg"] = segment["angle_deg"] if segment else ""
                fields[f"discharge_{cut_name}_{side}_segment{index}_support"] = segment["fit_support"] if segment else ""
    return fields


def repose_pair_result_fields(measurement: dict) -> dict:
    """Return machine-readable pair means and individual face angles for four repose axes."""
    expected_axes = {0, 45, 90, 135}
    pairs = measurement.get("pair_profiles", [])
    fields = {}
    for pair in pairs:
        axis = int(round(float(pair["axis_angle_deg"])))
        if axis not in expected_axes:
            raise ValueError(f"unexpected repose diametral axis: {axis}")
        fields[f"repose_axis{axis}_deg"] = float(pair["mean_angle_deg"])
        fields[f"repose_axis{axis}_plus_deg"] = float(pair["positive_angle_deg"])
        fields[f"repose_axis{axis}_minus_deg"] = float(pair["negative_angle_deg"])
    if {int(key.split("axis", 1)[1].split("_", 1)[0]) for key in fields if key.endswith("_deg")} != expected_axes:
        raise ValueError("repose result must contain exactly four diametral pairs")
    return fields


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trial_json_ready(value):
    """Keep estimator arrays/diagnostics in the same JSON form as their reports."""
    if isinstance(value, dict):
        return {key:_trial_json_ready(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_trial_json_ready(item) for item in value]
    if hasattr(value, 'tolist'):
        return _trial_json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class GeometryDiagnostic(RuntimeError):
    """The solver completed but the resulting geometry does not define the requested angle."""


class MeasurementRejected(RuntimeError):
    """Identifica uma geometria de saída inadequada para leitura, apesar de o solver ter concluído."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class CalibrationController:
    def __init__(self, project_root: Path, run_root: Path | None = None, solver_command=None, parallel_trials_override: int | None = None):
        # Embedded QC JSON exceeds csv's 128 KiB default. Configure before
        # compatibility validation and retain any larger caller-supplied limit.
        csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
        self.project_root = Path(project_root).resolve()
        isolated_root = Path(__file__).resolve().parent.parent
        original_root = Path('/mnt/c/DEM/campanhas/versao_antiga_ced11_5milhoes_r2_090_20261029').resolve()
        candidate_run = Path(run_root or self.project_root / 'calibration_runs').resolve()
        if (self.project_root == original_root or self.project_root.is_relative_to(original_root)
                or candidate_run == original_root or candidate_run.is_relative_to(original_root)):
            raise ValueError('original campaign is read-only: refusing controller')
        if self.project_root == isolated_root and candidate_run != isolated_root / 'run':
            raise ValueError('isolated campaign requires its own run directory')
        self.config = load_configuration(self.project_root / "configuracao_otimizacao.txt")
        self._rolling_protocol = self.config.get('rolling_protocol')
        if 'rolling' in self.config.get('tests', []) and (self.project_root/'Ang Rolamento/in.rolamento').is_file():
            from optimizer.rolling_protocol import effective_protocol
            self._rolling_protocol = effective_protocol(self.project_root/'Ang Rolamento', self.config['rolling_protocol'])
        if parallel_trials_override is not None:
            configured_workers = int(self.config.get("parallel_trials", 1))
            parallel_trials_override = int(parallel_trials_override)
            if parallel_trials_override < 1:
                raise ValueError("parallel_trials override must be positive")
            if parallel_trials_override > configured_workers:
                raise ValueError(f"parallel_trials override={parallel_trials_override} exceeds configured ceiling ({configured_workers})")
            self.config["parallel_trials"] = parallel_trials_override
        self.run_root = Path(run_root or self.project_root / "calibration_runs").resolve()
        self._validate_run_compatibility()
        self.run_root.mkdir(parents=True, exist_ok=True)
        identity_path = self.run_root / 'campaign_identity.json'
        if not identity_path.exists():
            identity_path.write_text(json.dumps(self._campaign_identity(), indent=2), encoding='utf-8')
        self.solver_command = list(solver_command) if solver_command else None
        self._csv_lock = RLock()
        self._results_revision = 0
        self._console_lock = Lock()
        self._cohesion_lock = Lock()
        guard = self.config.get("cohesion_guard", {})
        cohesion_bounds = self.config.get("parameters", {}).get("CED11")
        self._cohesion_tracker = CohesionLimitTracker(
            float(cohesion_bounds["min"]),
            float(cohesion_bounds["max"]),
            max_probes=int(guard.get("max_boundary_probes", 8)),
        ) if guard.get("enabled", True) and cohesion_bounds else None
        self._boundary_search_report = None
        self._next_trial_id = 1
        self._attempts = {}

    @property
    def parameter_names(self):
        return tuple(self.config["parameters"])

    @property
    def tests(self):
        names = self.config.get('tests', ['discharge', 'repose'])
        if not names or len(set(names)) != len(names) or set(names) - {t[0] for t in TESTS}:
            raise ValueError('invalid configured tests')
        return tuple(next(t for t in TESTS if t[0] == name) for name in names)

    @property
    def response_scales(self):
        return {kind: self.config.get('response_scales', {}).get(kind, self.config.get('tolerance_deg', .5)) for kind, *_ in self.tests}

    def _decks(self):
        for kind, directory, name, snapshot in self.tests:
            if kind == 'rolling':
                yield kind, directory, 'in.fill_1s', 'write_restart filled_1s.restart'
            yield kind, directory, name, snapshot

    def _write_cohesion_report(self, stage: str = "running") -> None:
        if self._cohesion_tracker is None:
            return
        with self._cohesion_lock:
            report = self._cohesion_tracker.report()
            report.update({
                "status": stage,
                "interpretation": "A geometry diagnostic is evidence of a non-measurable response, not proof that cohesion alone caused it.",
                "boundary_search": self._boundary_search_report,
            })
            if self._boundary_search_report and self._boundary_search_report.get("status") == "complete":
                report["recommended_ceiling_CED11"] = self._boundary_search_report["recommended_ceiling_CED11"]
                report["lowest_problematic_CED11"] = self._boundary_search_report["lowest_problematic_CED11"]
                report["highest_problematic_CED11"] = self._boundary_search_report["highest_problematic_CED11"]
                report["highest_measurable_CED11"] = self._boundary_search_report["highest_measurable_CED11"]
            target = self.run_root / "cohesion_limits.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(target)

    def _observe_cohesion(self, value: float, trial_id: int, parameters: dict, measurable: bool, reason: str = "") -> None:
        if self._cohesion_tracker is None:
            return
        with self._cohesion_lock:
            if measurable:
                self._cohesion_tracker.observe_valid(value, trial_id, parameters)
            else:
                self._cohesion_tracker.observe_problematic(value, trial_id, parameters, reason)
        self._write_cohesion_report()

    def _solver_invocation(self, cwd: Path, input_name: str) -> list[str]:
        if self.solver_command is not None:
            return [*self.solver_command, "-in", input_name]
        solver = self.config.get("solver", "/usr/local/bin/liggghts")
        mpi_processes = int(self.config.get("mpi_processes", 1))
        unbuffered_solver = ["stdbuf", "-oL", "-eL", solver]
        if os.name == "nt":
            distro = self.config.get("wsl_distribution", "Ubuntu-24.04")
            linux_cwd = subprocess.check_output(["wsl.exe", "-d", distro, "--", "wslpath", "-a", str(cwd)], text=True).strip()
            if mpi_processes > 1:
                return ["wsl.exe", "-d", distro, "--cd", linux_cwd, "--", "mpirun", "--allow-run-as-root", "--oversubscribe", "-np", str(mpi_processes), *unbuffered_solver, "-in", input_name]
            return ["wsl.exe", "-d", distro, "--cd", linux_cwd, "--", *unbuffered_solver, "-in", input_name]
        if mpi_processes > 1:
            return ["mpirun", "-np", str(mpi_processes), *unbuffered_solver, "-in", input_name]
        return [*unbuffered_solver, "-in", input_name]

    def preflight(self) -> dict:
        problems = []
        for kind, directory, input_name, snapshot in self._decks():
            source = self.project_root / directory
            input_path = source / input_name
            if not input_path.is_file():
                problems.append(f"missing input: {input_path}")
                continue
            text = input_path.read_text(encoding="utf-8")
            for name in self.parameter_names:
                if not re.search(PARAMETER_LINE.format(name=re.escape(name)), text, re.MULTILINE):
                    problems.append(f"missing variable {name} in {input_path}")
            for name, expected in self.config.get("fixed_parameters", {}).items():
                fixed_pattern = re.compile(rf"(?m)^\s*variable\s+{re.escape(name)}\s+equal\s+(\S+)")
                fixed_values = fixed_pattern.findall(text)
                if len(fixed_values) != 1:
                    problems.append(f"missing or ambiguous fixed variable {name} in {input_path}")
                else:
                    try:
                        if not math.isclose(float(fixed_values[0]), float(expected), rel_tol=0.0, abs_tol=1e-12):
                            problems.append(f"fixed variable {name} is {fixed_values[0]}, expected {expected}, in {input_path}")
                    except ValueError:
                        problems.append(f"fixed variable {name} is not numeric in {input_path}")
            for seed_name, pattern in SEED_PATTERNS.items():
                if (kind == 'rolling' and input_name == 'in.rolamento') or (kind == 'repose' and 'ANSYS_REPOSE_10S_ONE_SHOT' in text):
                    pattern = re.compile(rf'(?m)^variable\s+{seed_name}_seed\s+index\s+(\d+)')
                if len(pattern.findall(text)) != 1:
                    problems.append(f"missing or ambiguous {seed_name} seed site in {input_path}")
            if snapshot not in text:
                problems.append(f"missing write_dump for {snapshot} in {input_path}")
            if kind == 'rolling':
                if input_name == 'in.rolamento':
                    from optimizer.rolling_protocol import validate_native_event
                    try:
                        validate_native_event(source, text)
                    except (ValueError, OSError) as exc:
                        problems.append(str(exc))
                    active = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())
                    population_change = (r'(?m)^\s*(?:delete_atoms|create_atoms|read_data|replicate|clear)\b'
                                         r'|^\s*fix\s+\S+\s+\S+\s+(?:insert/|pour\b|deposit\b)')
                    restarts = re.findall(r'(?m)^\s*read_restart\s+(\S+)', active)
                    if re.search(population_change, active) or restarts != ['filled_1s.restart']:
                        problems.append('rolling aggregate measurement requires unchanged particle population')
                elif not re.search(r'(?m)^thermo_modify\s+lost\s+error\b',text) or re.search(r'(?m)^\s*thermo_modify\s+lost\s+ignore',text):
                    problems.append('rolling fill must fail closed on lost particles')
                if text.index('thermo_style') > text.index('thermo_modify'):
                    problems.append('rolling thermo_style must precede thermo_modify')
                for mesh in ('base','copo'):
                    if not re.search(rf'(?m)^fix {mesh} .*meshes/{mesh}.stl type 2\b',text):
                        problems.append(f'rolling {mesh} must contact true material type 2')
                for prefix in ('sf','rf','CED'):
                    if not re.search(rf'(?m)^variable\s+{prefix}21\s+equal\s+\$\{{{prefix}12\}}',text):
                        problems.append('rolling symmetric type 2 matrix missing: '+prefix)
                for prop,prefix in [('coefficientFriction','sf'),('coefficientRollingFriction','rf'),('cohesionEnergyDensity','CED')]:
                    expected=f'{prop} peratomtypepair ${{natoms}} ${{{prefix}11}} ${{{prefix}12}}'
                    if expected not in text:
                        problems.append('rolling contact matrix mismatch: '+prop)
        mpi_processes = int(self.config.get("mpi_processes", self.config.get("mpi_processes_per_trial", 1)))
        ranks_per_trial = int(self.config.get("mpi_processes_per_trial", mpi_processes))
        if mpi_processes < 1 or ranks_per_trial != mpi_processes:
            problems.append(f"MPI ranks mismatch: mpi_processes={mpi_processes}, mpi_processes_per_trial={ranks_per_trial}")
        for _, directory, input_name, _ in self._decks():
            path = self.project_root / directory / input_name
            if not path.is_file():
                continue
            deck = path.read_text(encoding="utf-8")
            grid = re.search(r"(?m)^\s*processors\s+(\d+)\s+(\d+)\s+(\d+)\b", deck)
            if not grid or math.prod(map(int, grid.groups())) != mpi_processes:
                problems.append(f"processors grid in {directory}/{input_name} must multiply to {mpi_processes}")
        available_cpus = os.cpu_count()
        requested_workers = int(self.config.get("parallel_trials", 1))
        if available_cpus is not None and requested_workers > available_cpus:
            problems.append(f"parallel_trials={requested_workers} exceeds CPUs visible to Python/WSL ({available_cpus})")
        if problems:
            raise RuntimeError("preflight failed: " + "; ".join(problems))
        return {"status": "ok", "campaign": self.config["campaign"], "parameters": list(self.parameter_names), "parallel_trials": requested_workers, "available_cpus": available_cpus}

    def prepare_trial(self, trial_id: int, parameters: dict[str, float], seeds: dict[str, int] | None = None) -> Path:
        workspace = self.run_root / f"trial_{trial_id:04d}"
        if workspace.exists():
            raise FileExistsError(f"trial workspace already exists: {workspace}")
        if set(parameters) != set(self.parameter_names) or not self._is_within_active_bounds(parameters):
            raise ValueError('trial requires exactly the configured finite factors within bounds')
        seeds = seeds or self._trial_seeds(trial_id)
        if set(seeds) != set(SEED_PATTERNS) or len(set(seeds.values())) != 2 or not all(is_valid_liggghts_seed(s) for s in seeds.values()):
            raise ValueError('distinct valid prime distribution/insertion seeds required')
        protocol = None
        if 'rolling' in self.config.get('tests', []):
            from optimizer.rolling_protocol import effective_protocol
            protocol = effective_protocol(self.project_root/'Ang Rolamento', self.config['rolling_protocol'])
            if protocol != self._rolling_protocol:
                raise ValueError('rolling mesh/scale changed since campaign identity; use a fresh run directory')
        workspace.mkdir(parents=True)
        for _, directory, _, _ in self.tests:
            (workspace / directory).mkdir()
            shutil.copytree(self.project_root / directory / 'meshes', workspace / directory / 'meshes')
        for _, directory, input_name, _ in self._decks():
            shutil.copy2(self.project_root / directory / input_name, workspace / directory / input_name)
            rewrite_input_parameters(workspace / directory / input_name, parameters)
            rewrite_input_seeds(workspace / directory / input_name, seeds)
            if directory == 'Ang Repouso' and 'ANSYS_REPOSE_10S_ONE_SHOT' in (workspace/directory/input_name).read_text():
                from optimizer.repose_ansys import verify_overlay
                verify_overlay(self.project_root)
                for asset in ('initial_particles.data', 'initial_audit.json'):
                    shutil.copy2(self.project_root/directory/asset, workspace/directory/asset)
            if directory == 'Ang Rolamento' and input_name == 'in.rolamento':
                from optimizer.rolling_protocol import event_includes, tilt_commands, validate_native_event
                deck = workspace/directory/input_name
                text = deck.read_text(encoding='utf8')
                start = text.index('log post/rolling_thermo.log')
                deck.write_text(text[:start]+tilt_commands(protocol)+'\n', encoding='utf8')
                for name, text in event_includes(protocol['version']).items():
                    (workspace/directory/name).write_text(text+'\n', encoding='utf8')
                validate_native_event(workspace/directory, deck.read_text(encoding='utf8'))
                from optimizer.rolling_retention import initialize_retention
                initialize_retention(workspace, protocol['version'])
        (workspace / "parameters.json").write_text(json.dumps({"parameters": parameters, "fixed_parameters": self.config.get("fixed_parameters", {}), "seeds": seeds,
            **({'rolling_protocol':protocol} if protocol else {})}, indent=2), encoding="utf-8")
        return workspace

    def _campaign_identity(self) -> dict:
        return {
            'schema_version': 1,
            'calibration_name': self.config.get('calibration_name'),
            'parameters': [[name, self.config['parameters'][name]] for name in self.parameter_names],
            'fixed_parameters': self.config.get('fixed_parameters', {}),
            'tests': self.config.get('tests', []),
            **({'rolling_protocol': self._rolling_protocol}
               if 'rolling' in self.config.get('tests', []) else {}),
            **({'sequential_protocol': {
                'charge_revision': self.config['charge_revision'],
                'assay_protocol_sha256': self.config['assay_protocol_sha256'],
            }} if 'sequential_experiment' in self.config else {}),
        }

    def _validate_run_compatibility(self) -> None:
        """Fail before writing: a historical five-factor campaign is not six-factor evidence."""
        identity = self.run_root / 'campaign_identity.json'
        if ('rolling' in self.config.get('tests', []) and not identity.exists() and
                (any((self.run_root/name).exists() for name in ('results.csv','scores.csv','plan.json','offline_plan.json','doe_manifest_original.json'))
                 or any(self.run_root.glob('trial_*')))):
            raise ValueError('rolling campaign protocol identity missing; use a fresh run directory')
        if identity.is_file() and json.loads(identity.read_text(encoding='utf-8')) != self._campaign_identity():
            raise ValueError('incompatible campaign identity; use a fresh run directory')
        for name in ('results.csv', 'scores.csv'):
            path = self.run_root / name
            if not path.is_file():
                continue
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if not set(self.parameter_names).issubset(reader.fieldnames or []):
                    raise ValueError(f'incompatible factor schema in {name}; use a fresh run directory')
                if any(not self._is_within_active_bounds(row) for row in reader):
                    raise ValueError(f'incompatible factor values in {name}; finite complete points are required')
        for name in ('plan.json', 'offline_plan.json', 'doe_manifest_original.json'):
            path = self.run_root / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            saved_config = data.get('configuration', {})
            if saved_config and saved_config.get('parameters') != self.config['parameters']:
                raise ValueError(f'incompatible factor configuration in {name}')
            points = data.get('points')
            if not isinstance(points, list) or any(set(p) != set(self.parameter_names) or not self._is_within_active_bounds(p) for p in points):
                raise ValueError(f'incompatible factor points in {name}; use a fresh run directory')
        for path in self.run_root.glob('trial_*/parameters.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            point = data.get('parameters', {})
            if set(point) != set(self.parameter_names) or not self._is_within_active_bounds(point):
                raise ValueError(f'incompatible factor provenance in {path.name}; preserve the old run separately')
        diagnostics = self.run_root / 'diagnostics.csv'
        if diagnostics.is_file():
            with diagnostics.open(newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    if row.get('parameters_json'):
                        point = json.loads(row['parameters_json'])
                        if set(point) != set(self.parameter_names) or not self._is_within_active_bounds(point):
                            raise ValueError('incompatible factor provenance in diagnostics.csv')
        status = self.run_root / 'status.csv'
        if not identity.exists() and status.is_file():
            with status.open(newline='', encoding='utf-8') as handle:
                if next(csv.DictReader(handle), None) is not None:
                    raise ValueError('incompatible campaign identity missing for status.csv; preserve this run and choose a new run-root')

    def _append_csv(self, filename: str, fields, row: dict) -> None:
        path = self.run_root / filename
        with self._csv_lock:
            write_header = not path.exists()
            if not write_header:
                with path.open(newline='', encoding='utf-8') as prior:
                    if next(csv.reader(prior), None) != list(fields):
                        raise ValueError(f'incompatible CSV schema in {filename}; refusing to append')
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in fields})
                handle.flush()
                os.fsync(handle.fileno())
                if filename == 'results.csv':
                    self._results_revision = getattr(self, '_results_revision', 0)+1

    def _record_status(self, **row) -> None:
        self._append_csv("status.csv", STATUS_FIELDS, row)

    def _run_input(self, workspace: Path, directory: str, input_name: str, stage: str) -> None:
        if self.config.get('fresh_finite_independent') is True:
            expected_root = (self.project_root / 'run').resolve()
            if self.run_root.resolve() != expected_root or not Path(workspace).resolve().is_relative_to(expected_root):
                raise RuntimeError('HOLD: fresh solver workspace outside independent run root')
        from optimizer.repose_ansys import require_scientific_protocol_compatibility
        require_scientific_protocol_compatibility(self.project_root, self.config)
        from optimizer.launch_authorization import require_launch_authorization
        approval_file = self.project_root / 'launch_authorization.json'
        if approval_file.exists() and json.loads(approval_file.read_text()).get('scope') == 'repose-single-validation':
            from optimizer.single_repose_validation import check_single_invocation
            check_single_invocation(self.project_root, workspace, directory, input_name,
                                    self._solver_invocation(workspace / directory, input_name))
        require_launch_authorization(self.project_root, self.config)
        if not self.config.get("simulation_launch_enabled", False):
            raise RuntimeError("simulation launching disabled: offline work only")
        trial_dir = workspace / directory
        command = self._solver_invocation(trial_dir, input_name)
        log_path = workspace / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=trial_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"{stage} solver failed with exit code {returncode}; see {log_path}")

    @staticmethod
    def _mesh_bounds_for_snapshot(snapshot_path: Path) -> dict[str, tuple[float, float]]:
        """Get X/Z bounds from one legacy emitted STL or derive from the trial's source mesh and scale."""
        post_dir = Path(snapshot_path).parent
        candidates = []
        for path in post_dir.glob("box_*.stl"):
            match = re.fullmatch(r"box_(\d+)\.stl", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            _, latest = max(candidates, key=lambda item: item[0])
            bounds = parse_ascii_stl_bounds(latest.read_text(encoding="utf-8", errors="replace"))
            return {axis: bounds[axis] for axis in ("x", "z")}

        test_dir = Path(snapshot_path).parents[1]
        deck_path = test_dir / "in.deslizamento"
        if not deck_path.is_file():
            raise RuntimeError(f"discharge deck missing beside measurement snapshot: {deck_path}")
        deck = deck_path.read_text(encoding="utf-8", errors="replace")
        box_fix = re.search(
            r"(?im)^\s*fix\s+box\s+\S+\s+mesh/surface\s+file\s+(\S+)\s+.*?\bscale\s+(\S+)",
            deck,
        )
        if not box_fix:
            raise RuntimeError(f"could not find scaled box mesh declaration in {deck_path}")

        definitions = {}
        for line in deck.splitlines():
            line = line.split("#", 1)[0]
            match = re.match(r"\s*variable\s+(\w+)\s+(?:equal|index)\s+(.+?)\s*$", line, re.IGNORECASE)
            if match:
                definitions[match.group(1)] = match.group(2).strip()

        def evaluate(expression: str, stack: tuple[str, ...] = ()) -> float:
            expression = re.sub(r"\$\((.*?)\)", r"\1", expression.strip())
            tree = ast.parse(expression, mode="eval")

            def visit(node) -> float:
                if isinstance(node, ast.Expression):
                    return visit(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.Name) and node.id.startswith("v_"):
                    name = node.id[2:]
                    if name in stack or name not in definitions:
                        raise ValueError(f"unknown or cyclic LIGGGHTS variable {node.id}")
                    return evaluate(definitions[name], stack + (name,))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    value = visit(node.operand)
                    return value if isinstance(node.op, ast.UAdd) else -value
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    left, right = visit(node.left), visit(node.right)
                    if isinstance(node.op, ast.Add): return left + right
                    if isinstance(node.op, ast.Sub): return left - right
                    if isinstance(node.op, ast.Mult): return left * right
                    if isinstance(node.op, ast.Div): return left / right
                    return left ** right
                raise ValueError(f"unsupported LIGGGHTS scale expression: {expression}")

            return visit(tree)

        source_mesh = (test_dir / box_fix.group(1)).resolve()
        scale = evaluate(box_fix.group(2))
        if not source_mesh.is_file() or not math.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"invalid source box mesh or scale: {source_mesh}, scale={scale}")
        bounds = parse_ascii_stl_bounds(source_mesh.read_text(encoding="utf-8", errors="replace"))
        bounds = {axis: (low * scale, high * scale) for axis, (low, high) in bounds.items()}
        return {axis: bounds[axis] for axis in ("x", "z")}

    @staticmethod
    def _nbr_ratio_from_deck(deck_path: Path) -> float:
        """Evaluate the trial-local LIGGGHTS nbrratio expression without executing deck code."""
        if not deck_path.is_file():
            raise RuntimeError(f"NBR deck missing: {deck_path}")
        definitions = {}
        for line in deck_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0]
            match = re.match(r"\s*variable\s+(\w+)\s+(?:equal|index)\s+(.+?)\s*$", line, re.IGNORECASE)
            if match:
                definitions[match.group(1)] = match.group(2).strip()

        def evaluate(expression: str, stack: tuple[str, ...] = ()) -> float:
            expression = re.sub(r"\$\((.*?)\)", r"\1", expression.strip())
            tree = ast.parse(expression, mode="eval")

            def visit(node) -> float:
                if isinstance(node, ast.Expression):
                    return visit(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.Name) and node.id.startswith("v_"):
                    name = node.id[2:]
                    if name in stack or name not in definitions:
                        raise ValueError(f"unknown or cyclic LIGGGHTS variable {node.id}")
                    return evaluate(definitions[name], stack + (name,))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    value = visit(node.operand)
                    return value if isinstance(node.op, ast.UAdd) else -value
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    left, right = visit(node.left), visit(node.right)
                    if isinstance(node.op, ast.Add): return left + right
                    if isinstance(node.op, ast.Sub): return left - right
                    if isinstance(node.op, ast.Mult): return left * right
                    if isinstance(node.op, ast.Div): return left / right
                    return left ** right
                raise ValueError(f"unsupported LIGGGHTS NBR expression: {expression}")

            return visit(tree)

        ratio = evaluate("v_nbrratio")
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"invalid nbrratio in {deck_path}: {ratio}")
        return ratio

    @staticmethod
    def _nbr_note_for_snapshot(snapshot_path: Path) -> str:
        """Record the NBR radius and ratio expressions from this trial's own input deck."""
        case_dir = snapshot_path.parents[1]
        deck_name = "in.repouso" if case_dir.name.lower().startswith("ang repouso") else "in.deslizamento"
        deck = case_dir / deck_name
        if not deck.is_file():
            return "NBR scaling: deck unavailable"
        found = {}
        for line in deck.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"\s*variable\s+(radiusnbr|nbrratio)\s+(?:equal|index)\s+([^#]+)", line, re.IGNORECASE)
            if match:
                found[match.group(1).lower()] = match.group(2).strip()
        return "NBR deck values: " + "; ".join(f"{name}={found[name]}" for name in ("radiusnbr", "nbrratio") if name in found) if found else "NBR scaling values not found in deck"

    def _measure_snapshot(self, path: Path, kind: str, output: Path | None = None) -> dict:
        from optimizer.protocol_eligibility import final_repose_snapshot
        if kind == 'repose':
            path = final_repose_snapshot(path)
        from optimizer.rocky_io import measure_snapshot, write_json
        from optimizer.response_policy import POLICY_VERSION, apply_response_policy
        import hashlib
        run_key = self.run_root.name + '__' + hashlib.sha256(str(self.run_root).encode('utf8')).hexdigest()[:10]
        trial_name = path.parents[1].name if kind == 'rolling' else path.parents[2].name
        target = output or self.project_root / "auditorias" / run_key / trial_name / kind
        if kind == 'rolling':
            from optimizer.rolling_measurement import measure_rolling
            result = measure_rolling(path)
            from optimizer.material_metadata import properties_from_deck
            try:
                result['material_properties'] = properties_from_deck(path, kind)
            except OSError as exc:
                result['material_properties'] = {'unavailable_reason': str(exc)}
            target.mkdir(parents=True, exist_ok=True)
            result['optimization_response'] = dict(
                response_policy=(f'rolling-total-speed-first-crossing-{result["protocol_version"].lower()}' if result.get('protocol_version') in ('V3', 'V4', 'V5', 'V6_LIFT_HEIGHT_3S', 'V7_PRESENT_POPULATION')
                                 else 'rolling-collective-motion-v1'),
                adopted_angle_deg=result['response_angle_deg'], adopted_status=result['status'],
                measurement_valid=result['status'] == 'measured', assignment_reason=None)
            from optimizer.rolling_visuals import render_rolling_snapshot
            result['snapshot'] = render_rolling_snapshot(path, result)
            result['visualization_status'] = result['snapshot']['status']
            path.mkdir(parents=True, exist_ok=True)
            provenance = path.parent.parent/'parameters.json'
            if provenance.is_file():
                result['rolling_protocol'] = json.loads(provenance.read_text(encoding='utf8')).get('rolling_protocol')
            write_json(path / 'rolling_measurement.json', result)
            write_json(target / 'Rolling_Report.json', result)
            if result['status'] != 'measured':
                raise MeasurementRejected(f"rolling: {result['reason']}; see {target}", result)
            return result
        result = measure_snapshot(path, kind, target, self.config["measurement"])
        policy = self.config['measurement'].get('response_policy')
        if policy is not None and policy != POLICY_VERSION:
            raise ValueError('unsupported response policy: ' + str(policy))
        raw = {'status': 'processed', 'particles': result['particle_count'],
               'new_status': result['status'], 'new_angle_deg': result['response_angle_deg'],
               'raw_reason': result.get('reason'), 'source_sha256': result['source_sha256'],
               'geometry_diagnosis': result.get('geometry_diagnosis'),
               'minimum_fit_r_squared': result.get('minimum_fit_r_squared'),
               'material_properties': result.get('material_properties')}
        response = apply_response_policy(raw)
        if policy is None and response['adopted_status'] != 'measured':
            response.update(adopted_angle_deg=None, adopted_status='unresolved', assignment_reason=None)
        response['response_policy'] = policy
        write_json(target / 'Response_Policy.json', response)
        result['optimization_response'] = response
        if result['status'] in ('invalid_data','error','failed'):
            raise ValueError(f"{kind} invalid artifacts: {result.get('reason')}; see {target}")
        if response['adopted_angle_deg'] is None:
            raise MeasurementRejected(f"{kind}: {result.get('reason', 'unmeasurable profile')}; see {target}", result)
        return result

    def reprocess_pair(self, source_workspace: Path, output: Path) -> dict:
        """Measure all configured SAVED cases; never run/prepare a trial."""
        from optimizer.rocky_io import measure_snapshot, write_json
        source_workspace, output = Path(source_workspace).resolve(), Path(output).resolve()
        if output.is_relative_to(source_workspace):
            raise ValueError("offline output must be outside the archived source trial")
        responses = {}
        for kind, directory, _, relative in self.tests:
            try:
                if kind == 'rolling':
                    from optimizer.rolling_measurement import measure_rolling
                    result = measure_rolling(source_workspace / directory / relative)
                    (output / kind).mkdir(parents=True, exist_ok=True)
                    write_json(output / kind / 'Rolling_Report.json', result)
                    responses[kind] = dict(status=result['status'], angle_deg=result['response_angle_deg'],
                        reason=result['reason'], source_sha256=result['source_sha256'],
                        report=str(output / kind / 'Rolling_Report.json'), qc=result)
                    continue
                result = measure_snapshot(source_workspace / directory / relative, kind,
                                          output / kind, self.config["measurement"])
                responses[kind] = {"status": result["status"],
                    "angle_deg": result["response_angle_deg"], "reason": result.get("reason"),
                    "source_sha256": result["source_sha256"],
                    "report": str(output / kind / (result["kind"]+"_Report.json"))}
            except (ValueError, OSError) as exc:
                responses[kind] = {"status":"invalid_data", "angle_deg":None,
                                   "reason":f"{type(exc).__name__}: {exc}"}
        complete = all(x["status"] == "measured" for x in responses.values())
        summary = {"source_workspace":str(source_workspace), "responses":responses,
                   "status":"measured" if complete else "incomplete_measurement",
                   "method":self.config["measurement"]["version"],
                   "offline_only":True, "real_dem_started":False,
                   "equivalence_to_ansys_verified":False}
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "Pair_Report.json", summary)
        return summary

    def execute_trial(self, trial_id: int, parameters: dict[str, float], phase: str = "doe", seeds: dict[str, int] | None = None) -> dict:
        from optimizer.launch_authorization import require_launch_authorization
        require_launch_authorization(self.project_root, self.config)
        if not self.config.get("simulation_launch_enabled", False):
            raise RuntimeError("simulation launching disabled: offline Rocky measurement work only")
        started = _utc_now()
        seeds = seeds or self._trial_seeds(trial_id)
        workspace = self.prepare_trial(trial_id, parameters, seeds)
        base = {"trial_id": trial_id, "phase": phase, "workspace": str(workspace), "started_at": started}
        measured = {}
        self._record_status(**base, status="running", stage="prepare", message="configured independent cases", finished_at="")
        try:
            for kind, directory, input_name, relative in self.tests:
                if kind == 'rolling':
                    self._record_status(**base, status='running', stage='rolling_fill', message='fresh fill to 1 s; settle/lift jointly through 10 s', finished_at='')
                    self._run_input(workspace, directory, 'in.fill_1s', 'rolling_fill')
                self._record_status(**base, status='running', stage=kind, message='executing '+kind, finished_at='')
                self._run_input(workspace, directory, input_name, kind)
            for kind, directory, _, relative in self.tests:
                try:
                    measured[kind] = self._measure_snapshot(workspace / directory / relative, kind)
                except MeasurementRejected as exc:
                    if exc.result is None:
                        raise  # Legacy exceptions without a payload are not physical evidence.
                    measured[kind] = exc.result
                if measured[kind]['status'] not in ('measured','unmeasurable','inconclusive','censored'):
                    raise ValueError(f"{kind}: invalid measurement artifacts: {measured[kind].get('reason')}")
                (workspace/'trial.json').write_text(json.dumps(_trial_json_ready(dict(base, status='running', responses=measured)),
                    ensure_ascii=False, allow_nan=False, indent=2), encoding='utf8')
            result = {**self.config.get("fixed_parameters", {}), **parameters,
                      "trial_id":trial_id, "phase":phase, "status":"complete", "workspace":str(workspace),
                      "distribution_seed":seeds["distribution"], "insertion_seed":seeds["insertion"],
                      "measurement_method":self.config["measurement"]["version"],
                      "discharge_response_convention":self.config["measurement"]["daor"]["response_convention"],
                      "equivalence_to_ansys_verified":False,
                      "discharge_left_deg":measured.get('discharge', {}).get('left_angle_deg'),
                      "discharge_right_deg":measured.get('discharge', {}).get('right_angle_deg'),
                      "discharge_paired_mean_deg":measured.get('discharge', {}).get('paired_mean_angle_deg'),
                      "discharge_asymmetry_deg":measured.get('discharge', {}).get('asymmetry_deg')}
            for kind in measured:
                raw = measured[kind]
                response = raw['optimization_response']
                result['response_policy'] = response['response_policy']
                result[kind+'_raw_deg'] = raw['response_angle_deg']
                result[kind+'_raw_status'] = raw['status']
                result[kind+'_raw_reason'] = raw.get('reason')
                result[kind+'_response_status'] = response['adopted_status']
                result[kind+'_measurement_valid'] = response['measurement_valid']
                result[kind+'_assignment_reason'] = response['assignment_reason']
                result[kind+"_deg"] = response['adopted_angle_deg']
                for suffix in ('physical_outcome','physical_outcome_verified','physical_outcome_evidence'):
                    result[kind+'_'+suffix] = response.get(suffix, raw.get(suffix))
                from optimizer.protocol_eligibility import interpret_for_controller, overlay_config, record_fresh_repose
                if kind == 'repose' and (overlay_config(self.project_root) or {}).get('schema_version') == 2:
                    record_fresh_repose(self, workspace, raw)
                result[kind+"_error"] = interpret_for_controller(self, result, kind, self.config['target_'+kind+'_deg'])['error_deg']
                if kind == 'rolling':
                    result['rolling_qc_json'] = json.dumps({k:v for k,v in raw.items() if k != 'optimization_response'}, allow_nan=False)
            # Only execute_trial owns fresh outputs. All frame consumers above
            # have finished; standalone readers and reprocess_pair never prune.
            if 'rolling' in measured:
                from optimizer.rolling_retention import finalize_rolling_trial
                measured['rolling']['retention'] = finalize_rolling_trial(
                    workspace/'Ang Rolamento/post', trial_root=workspace)
                result['rolling_qc_json'] = json.dumps(
                    {k:v for k,v in measured['rolling'].items() if k != 'optimization_response'}, allow_nan=False)
            result['response_policy'] = self.config['measurement'].get('response_policy')
            # Score the new row before its first append; never rewrite historical rows.
            from optimizer.adaptive_refinement import score_rows
            scored = score_rows(self, [result])[0]
            result['score'] = scored['score']
            result['accepted'] = self.responses_accepted(result)
            (workspace/'trial.json').write_text(json.dumps(_trial_json_ready(dict(result, responses=measured)),
                ensure_ascii=False, allow_nan=False, indent=2), encoding='utf8')
            self._append_results(result)
            summary = '; '.join(kind + ': ' + measured[kind]['optimization_response']['adopted_status'] for kind in measured)
            self._record_status(**base, status="complete", stage="measure", message=summary, finished_at=_utc_now())
            return result
        except MeasurementRejected as exc:
            self._record_status(**base, status="rejected", stage="measurement", message=str(exc), finished_at=_utc_now())
            raise
        except Exception as exc:
            (workspace/'trial.json').write_text(json.dumps(_trial_json_ready(dict(base, status='failed', error=str(exc), responses=measured)),
                ensure_ascii=False, allow_nan=False, indent=2), encoding='utf8')
            self._record_status(**base, status="failed", stage="failed", message=str(exc), finished_at=_utc_now())
            raise

    @staticmethod
    def _result_csv_row(result):
        encoded = dict(result)
        for kind in ('discharge','repose','rolling'):
            field = kind+'_physical_outcome_evidence'
            value = result.get(field)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict,list)): value = parsed
                except ValueError:
                    pass
            encoded[field] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return encoded

    def _append_results(self, result):
        self._append_csv('results.csv', RESULT_FIELDS, self._result_csv_row(result))

    @staticmethod
    def _restore_trial_from_csv(row):
        from optimizer.adaptive_outcomes import strict_bool, structured_evidence
        restored = dict(row)
        for kind in ('discharge','repose','rolling'):
            verified, evidence = kind+'_physical_outcome_verified', kind+'_physical_outcome_evidence'
            restored[verified] = strict_bool(row.get(verified))
            restored[evidence] = structured_evidence(row.get(evidence))
            if restored[evidence] is None and restored[verified] is True:
                restored[verified] = None
        return restored

    def _load_results_csv(self):
        path = self.run_root/'results.csv'
        if not path.is_file():
            return []
        with path.open(newline='', encoding='utf8') as handle:
            return [self._restore_trial_from_csv(row) for row in csv.DictReader(handle)]

    def score_completed_results(self) -> list[dict]:
        with self._csv_lock:
            return self._score_completed_results_unlocked()

    def _score_completed_results_unlocked(self) -> list[dict]:
        self._validate_run_compatibility()
        path = self.run_root / "results.csv"
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        metrics = [kind for kind, *_ in self.tests]
        weights = {kind: self.config['weights'][kind] for kind in metrics}
        weighted_normalized_score(dict.fromkeys(metrics, 0.), self.response_scales, weights)
        for row in rows:
            row['accepted'] = False
            row['score'] = ''
            if not self._valid_responses(row):
                continue
            errors = {metric: abs(float(row[metric+'_deg']) - self.config['target_'+metric+'_deg']) for metric in metrics}
            row.update({metric+'_error': value for metric,value in errors.items()})
            row['score'] = weighted_normalized_score(errors, self.response_scales, weights)
            row['accepted'] = self.responses_accepted(row)
        temporary=path.with_name('.results.scored.tmp')
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        from optimizer.protocol_eligibility import overlay_config
        schema2=(overlay_config(self.project_root) or {}).get('schema_version') == 2
        os.replace(temporary, path.with_name('results_recalculated.csv') if schema2 else path)
        if os.name != 'nt':
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        self._results_revision = getattr(self, '_results_revision', 0)+1
        return rows

    def _valid_responses(self, row):
        from optimizer.protocol_eligibility import interpret_for_controller
        return all(interpret_for_controller(self, row, kind)['regression_included'] for kind, *_ in self.tests)

    def responses_accepted(self, row):
        return self._valid_responses(row) and all(
            abs(float(row[kind+'_deg'])-self.config['target_'+kind+'_deg']) <= self.response_scales[kind] + 1e-9
            for kind, *_ in self.tests)

    def _trial_seeds(self, trial_id: int) -> dict[str, int]:
        rng = random.Random(int(self.config.get("random_seed", 1)) + trial_id * 1009)
        distribution = next_liggghts_prime(rng.randrange(10_001, 899_000_000))
        insertion = next_liggghts_prime(rng.randrange(10_001, 899_000_000))
        if insertion == distribution:
            insertion = next_liggghts_prime(insertion + 2)
        return {"distribution": distribution, "insertion": insertion}

    def _point_key(self, parameters: dict[str, float]) -> tuple[tuple[str, float], ...]:
        """Cria uma chave estável para retomar pontos DOE sem repetir parâmetros já encerrados."""
        return tuple((name, round(float(parameters[name]), 12)) for name in self.parameter_names)

    def _is_within_active_bounds(self, parameters: dict[str, float]) -> bool:
        """Aceita somente pontos dentro dos limites ativos, inclusive nas extremidades."""
        try:
            return all(math.isfinite(float(parameters[name])) and float(spec["min"]) <= float(parameters[name]) <= float(spec["max"]) for name, spec in self.config["parameters"].items())
        except (KeyError, ValueError, TypeError):
            return False

    def _calibrate_cohesion_ceiling(self) -> None:
        """Bracket cohesion while holding every other configured factor constant."""
        if self._cohesion_tracker is None:
            return
        if self._boundary_search_report and self._boundary_search_report.get("status") == "complete":
            return
        prior = self.run_root / "cohesion_boundary_search.json"
        if prior.is_file():
            saved = json.loads(prior.read_text(encoding="utf-8"))
            if saved.get("status") == "complete":
                self._boundary_search_report = saved
                ceiling = saved.get("recommended_ceiling_CED11")
                if ceiling is not None:
                    self.config["parameters"]["CED11"]["max"] = float(ceiling)
                self._write_cohesion_report("boundary_search_complete")
                return
        bad = next((row for row in sorted(self._cohesion_tracker.problematic, key=lambda row: row["CED11"]) if "sf11" in row["parameters"] and "rf11" in row["parameters"]), None)
        if bad is None:
            self._boundary_search_report = {"status": "not_needed", "detail": "No cohesion-associated non-measurable geometry was observed."}
            self._write_cohesion_report("no_problem_observed")
            return
        p = bad["parameters"]
        lower = float(self.config["parameters"]["CED11"]["min"])
        upper = float(bad["CED11"])
        pair = {name: p[name] for name in self.parameter_names if name != 'CED11'}
        existing = [row for row in self._cohesion_tracker.valid if all(abs(row["parameters"].get(name, float("inf")) - value) <= 1e-10 for name, value in pair.items()) and lower <= row["CED11"] < upper]
        if existing:
            low_row = max(existing, key=lambda row: row["CED11"])
            low_value = float(low_row["CED11"])
        else:
            baseline = {**p, "CED11": lower}
            completed = self._run_points([baseline], "cohesion_boundary", self._next_trial_id)
            if not completed:
                self._boundary_search_report = {"status": "no_valid_lower_bracket", "problematic_CED11": upper, "baseline_CED11": lower, "parameters": pair}
                prior.write_text(json.dumps(self._boundary_search_report, indent=2), encoding="utf-8")
                self._write_cohesion_report("boundary_search_blocked")
                return
            low_value = lower
        search = CohesionLimitTracker(lower, upper, max_probes=int(self.config.get("cohesion_guard", {}).get("max_boundary_probes", 8)))
        search.observe_valid(low_value, 0, pair)
        search.observe_problematic(upper, int(bad["trial_id"]), pair, bad.get("reason", "geometry not measurable"))
        probes = []
        while (candidate := search.next_probe()) is not None:
            point = {**p, "CED11": candidate}
            before_bad_count = len(self._cohesion_tracker.problematic)
            completed = self._run_points([point], "cohesion_boundary", self._next_trial_id)
            if completed:
                search.observe_valid(candidate, int(completed[0]["trial_id"]), point)
                probes.append({"CED11": candidate, "status": "measurable", "trial_id": completed[0]["trial_id"]})
            elif len(self._cohesion_tracker.problematic) > before_bad_count:
                latest = self._cohesion_tracker.problematic[-1]
                search.observe_problematic(candidate, int(latest["trial_id"]), point, latest["reason"])
                probes.append({"CED11": candidate, "status": "not_measurable", "trial_id": latest["trial_id"]})
            else:
                probes.append({"CED11": candidate, "status": "rejected_or_solver_failure"})
                break
        ceiling = search.recommended_ceiling
        self.config["parameters"]["CED11"]["max"] = ceiling
        self._boundary_search_report = {
            "status": "complete",
            "parameters": pair,
            "initial_problematic_CED11": upper,
            "highest_measurable_CED11": search.highest_valid_below_problem,
            "lowest_problematic_CED11": search.lowest_problematic,
            "highest_problematic_CED11": search.highest_problematic,
            "recommended_ceiling_CED11": ceiling,
            "remaining_interval_CED11": search.lowest_problematic - ceiling if search.lowest_problematic is not None else None,
            "probes": probes,
            "interpretation": "The ceiling holds all non-cohesion factors fixed; it is an empirical guard, not a universal material law.",
        }
        prior.write_text(json.dumps(self._boundary_search_report, indent=2, sort_keys=True), encoding="utf-8")
        self._write_cohesion_report("boundary_search_complete")

    def _doe_backfill_batch_size(self, completed_count: int) -> int:
        """Mantém um lote cheio de contingência até atingir o mínimo de DOE completo."""
        minimum = int(self.config.get("minimum_doe_complete", self.config["campaign"]["doe_trials"]))
        if completed_count >= minimum and not self.config.get('directed_doe_backfill', {}).get('enabled'):
            return 0
        batch_size = int(self.config.get('doe_contingency_batch_size', self.config.get('parallel_trials', 1)))
        # Caller invokes this only after a failed readiness gate. Reaching the
        # target row count is not sufficient if the quadratic matrix lacks rank.
        needed = min(minimum - completed_count, batch_size) if completed_count < minimum else batch_size
        cap = int(self.config.get('maximum_doe_attempts', 0) or 0)
        return min(needed, max(0, cap - self._doe_attempt_count())) if cap > 0 else needed

    def _doe_attempt_count(self) -> int:
        """Count every persisted reserved/failed/completed ID exactly once."""
        phases = {}
        for name in ('status.csv', 'results.csv', 'diagnostics.csv'):
            path = self.run_root/name
            if path.is_file():
                with path.open(newline='', encoding='utf8') as handle:
                    for row in csv.DictReader(handle):
                        if row.get('phase'):
                            tid = int(row['trial_id'])
                            phase = row['phase']
                            if tid in phases and phases[tid] != phase:
                                raise RuntimeError('ambiguous phase in attempt count')
                            phases[tid] = phase
        journal = self.run_root/'doe_schedule.jsonl'
        if journal.exists():
            for line in journal.read_text(encoding='utf8').splitlines():
                event = json.loads(line)
                tid = int(event['trial_id'])
                if tid in phases and phases[tid] != 'doe':
                    raise RuntimeError('ambiguous DOE reservation phase')
                phases[tid] = 'doe'
        for directory in self.run_root.glob('trial_*'):
            if directory.is_dir() and directory.name[6:].isdigit():
                tid = int(directory.name[6:])
                if tid not in phases:
                    meta = directory/'parameters.json'
                    data = json.loads(meta.read_text()) if meta.exists() else {}
                    phases[tid] = data.get('phase', 'doe')
        return sum(phase == 'doe' for phase in phases.values())

    def _doe_is_ready(self, scored_rows, *, write_report=False) -> bool:
        """At the hard cap, a smaller DOE needs a full-rank quadratic design."""
        from optimizer.protocol_eligibility import overlay_config, interpret_for_controller
        schema2=(overlay_config(self.project_root) or {}).get('schema_version')==2
        if schema2 and 'sequential_experiment' in self.config and self.config.get('doe_attempt_multiplier') != 5:
            # Repose has a new protocol, not an erased DOE. Require real full-rank
            # support for EACH unchanged response; new repose seeds via maximin.
            import numpy as np
            terms=recommend_campaign_size(self.parameter_names)['quadratic_terms']
            support={}
            for kind in self.config['tests']:
                if kind=='repose':continue
                measured=[r for r in scored_rows if r.get('phase')=='doe' and
                    interpret_for_controller(self,r,kind)['regression_included']]
                rank=int(np.linalg.matrix_rank(np.asarray([self._quadratic_features(r) for r in measured]))) if measured else 0
                if len(measured)<terms or rank<terms:
                    raise ValueError('unchanged response DOE support insufficient: '+kind)
                support[kind]={'measured_rows':len(measured),'rank':rank}
            if not support:raise ValueError('no unchanged response DOE support')
            if write_report:
                (self.run_root/'doe_protocol_handoff.json').write_text(json.dumps(support,indent=2))
            return True
        valid = [r for r in scored_rows if r.get('phase') == 'doe' and self._valid_responses(r)]
        target = int(self.config.get('minimum_doe_complete', self.config['campaign']['doe_trials']))
        cap = int(self.config.get('maximum_doe_attempts', 0) or 0)
        attempts = self._doe_attempt_count()
        capped = cap > 0 and attempts >= cap
        if len(valid) < target and (not capped or self.config.get('doe_attempt_multiplier') == 5):
            return False
        import numpy as np
        terms = recommend_campaign_size(self.parameter_names)['quadratic_terms']
        matrix = np.asarray([self._quadratic_features(r) for r in valid], dtype=float)
        rank = int(np.linalg.matrix_rank(matrix)) if len(valid) else 0
        if len(valid) < terms or rank < terms:
            if capped:
                raise ValueError(f'DOE cap reached ({attempts}/{cap}); insufficient independent data: {len(valid)} valid rows, rank {rank}/{terms}')
            return False
        if write_report:
            report = dict(target_valid=target, valid_rows=len(valid), physical_attempts=attempts,
                          maximum_doe_attempts=cap, quadratic_terms=terms, rank=rank,
                          fallback_at_cap=capped and len(valid) < target)
            (self.run_root / 'doe_completion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return True

    def _resume_doe_state(self) -> tuple[set[tuple[tuple[str, float], ...]], dict[tuple[tuple[str, float], ...], int], int]:
        """Lê resultados e rejeições persistidos para retomar apenas pontos DOE pendentes."""
        completed = set()
        rejected_counts = {}
        maximum_trial_id = 0
        results_path = self.run_root / "results.csv"
        if results_path.is_file():
            with results_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("phase") == "doe" and row.get("status") == "complete":
                        completed.add(self._point_key(row))
                        maximum_trial_id = max(maximum_trial_id, int(row["trial_id"]))
        status_path = self.run_root / "status.csv"
        if status_path.is_file():
            with status_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    maximum_trial_id = max(maximum_trial_id, int(row["trial_id"]))
                    if row.get("phase") != "doe" or row.get("status") != "rejected":
                        continue
                    workspace = Path(os.path.abspath(row["workspace"]))
                    if not workspace.is_relative_to(self.run_root.resolve()):
                        raise ValueError('historical workspace outside isolated run')
                    workspace = workspace.resolve()
                    if not workspace.is_relative_to(self.run_root.resolve()):
                        raise ValueError('historical workspace resolves outside isolated run')
                    parameters_path = workspace / "parameters.json"
                    if not parameters_path.is_file():
                        continue
                    parameters = json.loads(parameters_path.read_text(encoding="utf-8")).get("parameters", {})
                    key = self._point_key(parameters)
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
        diagnostics_path = self.run_root / "diagnostics.csv"
        if diagnostics_path.is_file():
            with diagnostics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("phase") != "doe" or not row.get("parameters_json"):
                        continue
                    parameters = json.loads(row["parameters_json"])
                    completed.add(self._point_key(parameters))
                    maximum_trial_id = max(maximum_trial_id, int(row["trial_id"]))
        return completed, rejected_counts, maximum_trial_id + 1

    def _run_points(self, points: list[dict], phase: str, first_trial_id: int, initial_retries: dict | None = None) -> list[dict]:
        cap = self.config.get('maximum_doe_attempts')
        if phase == 'cohesion_boundary' and cap is not None:
            raise RuntimeError('extra cohesion probes disabled in capped campaigns')
        path = self.run_root / 'status.csv'
        persisted = []
        if path.exists():
            with path.open(newline='',encoding='utf8') as handle:
                persisted = list(csv.DictReader(handle))
        if phase not in self._attempts:
            self._attempts[phase] = set()
        self._attempts[phase].update(int(r['trial_id']) for r in persisted if r.get('phase') == phase)
        # Recovery-only: keep abandoned files/IDs without charging them twice.
        self._attempts[phase].difference_update(getattr(self, '_recovery_abandoned_ids', {}).get(phase, ()))
        if phase == 'doe' and (self.run_root / 'results.csv').is_file():
            with (self.run_root / 'results.csv').open(newline='', encoding='utf-8') as handle:
                self._attempts[phase].update(int(r['trial_id']) for r in csv.DictReader(handle) if r.get('phase') == phase)
        limit = int(cap) if phase == 'doe' and cap is not None else int(self.config['campaign'].get(phase+'_trials', len(points)))
        if points and len(self._attempts[phase]) >= limit:
            raise RuntimeError(f'{phase} stopped at {limit} attempts: insufficient accepted measurements; no forced fit')
        if not points: return []
        used_ids = {int(r['trial_id']) for r in persisted}
        used_ids.update(int(p.name[6:]) for p in self.run_root.glob('trial_*') if p.name[6:].isdigit())
        if used_ids and first_trial_id <= max(used_ids):
            raise FileExistsError('trial IDs must be fresh and greater than all persisted attempts')
        workers = min(len(points), int(self.config.get("parallel_trials", 1)))
        if workers < 1:
            raise ValueError("parallel_trials must be positive")
        results = []
        point_iterator = iter(points)
        next_trial_id = first_trial_id
        retry_limit = int(self.config.get("measurement_retry_limit", 2))
        retries = dict(initial_retries or {})
        active = {}
        dashboard_interval = 10
        maximum = self.config["campaign"]["max_trials"]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {}
            def submit(point):
                nonlocal next_trial_id
                if len(self._attempts[phase]) >= limit:
                    return
                trial_id = next_trial_id; next_trial_id += 1
                self._attempts[phase].add(trial_id)
                self._record_status(trial_id=trial_id, phase=phase, status='reserved', stage='queue',
                    message='attempt reserved before worker starts',
                    workspace=str(self.run_root / f'trial_{trial_id:04d}'), started_at=_utc_now(), finished_at='')
                active[trial_id] = {"phase": phase, "status": "preparing", "started_at": time.time(), "simulation_time_s": 0.0}
                pending[executor.submit(self.execute_trial, trial_id, point, phase, self._trial_seeds(trial_id))] = (trial_id, point)

            for _ in range(workers):
                try:
                    submit(next(point_iterator))
                except StopIteration:
                    break
            while pending:
                refresh_live_state(self.run_root, active)
                write_live_status_csv(self.run_root, active)
                render_dashboard(active, first_trial_id - 1 + len(results), maximum, interval=dashboard_interval)
                completed, _ = wait(pending, timeout=dashboard_interval, return_when=FIRST_COMPLETED)
                for future in completed:
                    trial_id, point = pending.pop(future)
                    try:
                        result = future.result()
                        results.append(result)
                        # A scoreable assigned value is not a measured material observation.
                        if "CED11" in point and all(result.get(kind + '_response_status') == 'measured' for kind, *_ in self.tests):
                            self._observe_cohesion(float(point["CED11"]), trial_id, point, measurable=True)
                        active.pop(trial_id, None)
                    except GeometryDiagnostic as exc:
                        active.pop(trial_id, None)
                        if "CED11" in point:
                            self._observe_cohesion(float(point["CED11"]), trial_id, point, measurable=False, reason=str(exc))
                        self._append_csv("diagnostics.csv", DIAGNOSTIC_FIELDS, {"trial_id": trial_id, "phase": phase, "status": "geometry_unmeasurable", "reason": str(exc), "parameters_json": json.dumps(point, sort_keys=True), "workspace": str(self.run_root / f"trial_{trial_id:04d}")})
                        try:
                            submit(next(point_iterator))
                        except StopIteration:
                            pass
                        continue
                    except MeasurementRejected:
                        active.pop(trial_id, None)
                        key = tuple(sorted((name, float(value)) for name, value in point.items()))
                        retries[key] = retries.get(key, 0) + 1
                        if retries[key] > retry_limit:
                            write_live_status_csv(self.run_root, active)
                            try:
                                submit(next(point_iterator))
                            except StopIteration:
                                pass
                            continue
                        submit(point)
                        continue
                    except Exception as exc:
                        active.pop(trial_id, None)
                        self._record_status(trial_id=trial_id, phase=phase, status="failed", stage="failed", message=str(exc), workspace=str(self.run_root / f"trial_{trial_id:04d}"), started_at="", finished_at=_utc_now())
                        try:
                            submit(next(point_iterator))
                        except StopIteration:
                            pass
                        continue
                    try:
                        submit(next(point_iterator))
                    except StopIteration:
                        continue
            refresh_live_state(self.run_root, active)
            write_live_status_csv(self.run_root, active)
            render_dashboard(active, first_trial_id - 1 + len(results), maximum, interval=dashboard_interval)
        self._next_trial_id = next_trial_id
        return results

    def _quadratic_features(self, point: dict[str, float]) -> list[float]:
        values = [(float(point[name]) - self.config["parameters"][name]["min"]) / (self.config["parameters"][name]["max"] - self.config["parameters"][name]["min"]) for name in self.parameter_names]
        return [1.0, *values, *(value * value for value in values), *(values[i] * values[j] for i in range(len(values)) for j in range(i + 1, len(values)))]

    @staticmethod
    def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
        augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
        for column in range(len(vector)):
            pivot = max(range(column, len(vector)), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) < 1e-12:
                raise ValueError("quadratic surrogate is rank deficient")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            divisor = augmented[column][column]
            augmented[column] = [value / divisor for value in augmented[column]]
            for row in range(len(vector)):
                if row != column:
                    factor = augmented[row][column]
                    augmented[row] = [value - factor * base for value, base in zip(augmented[row], augmented[column])]
        return [row[-1] for row in augmented]

    def nominate_refinement_candidates(self, completed_rows: list[dict]) -> list[dict]:
        rows = [row for row in completed_rows if self._valid_responses(row) and row.get("score", "") != "" and math.isfinite(float(row['score'])) and row.get("phase") != "cohesion_boundary" and self._is_within_active_bounds(row)]
        initial = {name: self.config["parameters"][name]["initial"] for name in self.parameter_names}
        terms = len(self._quadratic_features(initial))
        if len(rows) < terms:
            raise ValueError(f"quadratic surrogate needs at least {terms} scored trials")
        features = [self._quadratic_features(row) for row in rows]
        scores = [float(row["score"]) for row in rows]
        import numpy as np
        if np.linalg.matrix_rank(features) != terms:
            raise ValueError('quadratic surrogate is rank deficient')
        coefficients = np.linalg.lstsq(features, scores, rcond=None)[0]
        bounds = {name: (spec["min"], spec["max"]) for name, spec in self.config["parameters"].items()}
        pool = latin_hypercube(int(self.config.get("model_candidate_pool", 1500)), bounds, int(self.config.get("random_seed", 1)) + 7919)
        observed = {tuple(round(float(row[name]), 12) for name in self.parameter_names) for row in rows}
        selected = []
        for _, point in sorted(((sum(value * coefficient for value, coefficient in zip(self._quadratic_features(point), coefficients)), point) for point in pool), key=lambda item: item[0]):
            key = tuple(round(float(point[name]), 12) for name in self.parameter_names)
            if key not in observed:
                selected.append(point)
                observed.add(key)
            if len(selected) == self.config["campaign"]["refinement_trials"]:
                return selected
        raise RuntimeError("could not nominate enough distinct refinement candidates")

    def _append_doe_event(self, event):
        """Write-ahead, fsynced reservation/terminal journal. Never launch on failure."""
        path = self.run_root/'doe_schedule.jsonl'
        with path.open('a', encoding='utf8') as handle:
            handle.write(json.dumps(dict(event, recorded_at=_utc_now()), allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _restore_async_doe_state(self):
        """Reconcile legacy provenance and the journal; NEVER reexecute uncertain work.

        Resume hook: stop only the old controller, drain its workers, preserve all
        outputs, then call run_doe under the campaign lock. Nonterminal IDs require
        explicit external reconciliation, not automatic abandonment or retry.
        """
        records = {}
        journal_reserved, journal_terminal = set(), set()

        def merge(tid, phase=None, point=None, terminal=None):
            tid = int(tid)
            if tid < 1:
                raise RuntimeError('invalid persisted trial ID')
            rec = records.setdefault(tid, dict(parameters=None, phase=None, terminal=None))
            if phase:
                if rec['phase'] and rec['phase'] != phase:
                    raise RuntimeError('ambiguous trial phase for '+str(tid))
                rec['phase'] = phase
            if point is not None:
                if set(point) != set(self.parameter_names) or not self._is_within_active_bounds(point):
                    raise RuntimeError('invalid persisted point for '+str(tid))
                point = {n: float(point[n]) for n in self.parameter_names}
                if rec['parameters'] is not None and self._point_key(rec['parameters']) != self._point_key(point):
                    raise RuntimeError('ambiguous parameters for trial '+str(tid))
                rec['parameters'] = point
            if terminal:
                if rec['terminal'] and rec['terminal'] != terminal:
                    raise RuntimeError('ambiguous terminal outcome for trial '+str(tid))
                rec['terminal'] = terminal
            return rec

        journal = self.run_root/'doe_schedule.jsonl'
        if journal.exists():
            for line in journal.read_text(encoding='utf8').splitlines():
                try:
                    event = json.loads(line)
                    tid = int(event['trial_id'])
                    kind = event['event']
                except (ValueError, KeyError, TypeError) as exc:
                    raise RuntimeError('ambiguous DOE journal; reconcile before resume') from exc
                if kind == 'reserved':
                    if tid in journal_reserved:
                        raise RuntimeError('duplicate DOE reservation ID '+str(tid))
                    journal_reserved.add(tid)
                    merge(tid, 'doe', event['parameters'])
                elif kind in ('completed', 'failed', 'rejected'):
                    if tid not in journal_reserved or tid in journal_terminal:
                        raise RuntimeError('ambiguous DOE terminal event '+str(tid))
                    journal_terminal.add(tid)
                    merge(tid, 'doe', terminal='complete' if kind == 'completed' else kind)
                else:
                    raise RuntimeError('unknown DOE journal event')
        adaptive = self.run_root/'adaptive_proposals.jsonl'
        if adaptive.exists():
            for line in adaptive.read_text(encoding='utf8').splitlines():
                event = json.loads(line)
                kind = event.get('event')
                if kind == 'reserved':
                    merge(event['trial_id'], 'refinement', event['parameters'])
                elif kind in ('completed', 'failed'):
                    merge(event['trial_id'], 'refinement', terminal='complete' if kind == 'completed' else 'failed')
                elif kind == 'abandoned_batch':
                    for tid in event['trial_ids']:
                        merge(tid, 'refinement', terminal='failed')
        result_ids = set()
        for name in ('status.csv', 'results.csv', 'diagnostics.csv'):
            path = self.run_root/name
            if not path.exists():
                continue
            with path.open(newline='', encoding='utf8') as handle:
                for row in csv.DictReader(handle):
                    if name == 'results.csv':
                        tid = int(row['trial_id'])
                        if tid in result_ids:
                            raise RuntimeError('duplicate persisted result ID '+str(tid))
                        result_ids.add(tid)
                    status = row.get('status')
                    point = None
                    if name == 'results.csv':
                        point = {n: row[n] for n in self.parameter_names}
                    elif row.get('parameters_json'):
                        point = json.loads(row['parameters_json'])
                    terminal = status if status in ('complete', 'failed', 'rejected') else None
                    if name == 'diagnostics.csv' and status == 'geometry_unmeasurable':
                        terminal = 'rejected'
                    # A 'complete' status without a committed measurement row is
                    # not enough to recover scientific data automatically.
                    if name == 'status.csv' and terminal == 'complete':
                        terminal = None
                    merge(row['trial_id'], row.get('phase'), point, terminal)
        for directory in self.run_root.glob('trial_*'):
            if not directory.is_dir() or not directory.name[6:].isdigit():
                continue
            tid = int(directory.name[6:])
            rec = merge(tid)
            path = directory/'parameters.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf8'))
                merge(tid, data.get('phase'), data.get('parameters'))
            trial = directory/'trial.json'
            if trial.exists():
                data = json.loads(trial.read_text(encoding='utf8'))
                # Terminal failures need no invented measurement. Complete trials
                # require the result CSV commit (or explicit offline recovery).
                merge(tid, data.get('phase'), terminal=data.get('status') if data.get('status') in ('failed', 'rejected') else None)
            if rec['phase'] is None:
                rec['phase'] = 'doe'
        uncertain = [tid for tid, rec in records.items() if not rec['terminal']]
        if uncertain:
            raise RuntimeError('nonterminal/ambiguous trial reservations; drain and reconcile before resume: '+str(sorted(uncertain)))
        return records

    def _commit_doe_reservation(self, event, revision=None, pending=()):
        with self._csv_lock:
            if revision is not None and (revision != getattr(self, '_results_revision', 0)
                    or any(f.done() for f in pending)):
                return False
            self._append_doe_event(event)
            return True

    def initial_doe_points(self) -> list[dict]:
        """Keep the DOE size; replace LHS slot zero with configured initials."""
        bounds = {name: (spec['min'], spec['max']) for name, spec in self.config['parameters'].items()}
        points = latin_hypercube(self.config['campaign']['doe_trials'], bounds, self.config['random_seed'])
        policy = self.config.get('doe_initial_point_policy')
        if policy == 'configured_initial_replaces_first_lhs':
            initial = {name: float(spec['initial']) for name, spec in self.config['parameters'].items()}
            if not self._is_within_active_bounds(initial):
                raise ValueError('configured initial DOE point is outside active bounds')
            points[0] = initial
            if len({self._point_key(p) for p in points}) != len(points):
                raise ValueError('configured initial point duplicates an existing LHS point')
        elif policy is not None:
            raise ValueError('unknown DOE initial point policy: ' + str(policy))
        return points

    def _run_doe_async(self):
        records = self._restore_async_doe_state()
        self._next_trial_id = max(max(records, default=0)+1, int(self.config.get('first_trial_id', 1)))
        cap = int(self.config['maximum_doe_attempts'])
        attempts = {tid for tid, rec in records.items() if rec['phase'] == 'doe'}
        if len(attempts) > cap:
            raise RuntimeError('persisted DOE attempts exceed configured cap')
        if self.config.get('doe_attempt_multiplier') == 5 and len(records) > int(self.config['campaign']['max_trials']):
            raise RuntimeError('persisted attempts exceed campaign capacity')
        bounds = {n: (s['min'], s['max']) for n, s in self.config['parameters'].items()}
        manifest_path = self.run_root/'doe_manifest_original.json'
        if manifest_path.exists():
            original = json.loads(manifest_path.read_text(encoding='utf8'))['points']
        else:
            original = self.initial_doe_points()
            temp = manifest_path.with_suffix('.tmp')
            with temp.open('w', encoding='utf8') as handle:
                payload=dict(points=original, normal_doe_points=len(original), maximum_doe_attempts=cap)
                json.dump(payload, handle, allow_nan=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(manifest_path)
        if (len(original) != int(self.config['campaign']['doe_trials'])
                or any(set(p) != set(self.parameter_names) or not self._is_within_active_bounds(p) for p in original)
                or len({self._point_key(p) for p in original}) != len(original)):
            raise ValueError('invalid authoritative initial DOE manifest')
        if self.config.get('doe_execution_order') == 'configured_initial_then_manifest':
            expected_initial = {n: float(s['initial']) for n, s in self.config['parameters'].items()}
            if not original or self._point_key(original[0]) != self._point_key(expected_initial):
                raise ValueError('authoritative manifest must begin with configured initial point')
        attempted = [rec['parameters'] for rec in records.values() if rec['parameters'] is not None]
        observed = {self._point_key(p) for p in attempted}
        # Terminal failed/rejected initial points are attempted, never retried.
        initial_queue = [(i, p) for i, p in enumerate(original) if self._point_key(p) not in observed]
        execution_order = self.config.get('doe_execution_order', 'manifest')
        if execution_order not in ('manifest', 'CED11_then_CED12_ascending', 'configured_initial_then_manifest'):
            raise ValueError('unknown DOE execution order: '+str(execution_order))
        if execution_order == 'CED11_then_CED12_ascending':
            # Sort dispatch only: the authoritative six-factor LHS is unchanged.
            # CED11 is particle-particle cohesion; CED12 is only a tie-breaker.
            initial_queue.sort(key=lambda pair: (float(pair[1]['CED11']), float(pair[1]['CED12'])))
        workers = int(self.config.get('parallel_trials', 1))
        if workers < 1:
            raise ValueError('parallel_trials must be positive')
        pending, active = {}, {}
        stop_dispatch = False
        prior_seeds = [int(p.stem.rsplit('_', 1)[1]) for p in self.run_root.glob('doe_selection_audit_*.json') if p.stem.rsplit('_', 1)[1].isdigit()]
        selection_seed = max([int(self.config.get('random_seed', 1))+209759+self._next_trial_id-1]+prior_seeds)
        scored = self._load_results_csv()

        def ingest(ready):
            nonlocal stop_dispatch
            for future in sorted(ready, key=lambda f: pending[f][0]):
                tid, point, initial_index = pending.pop(future)
                outcome = 'completed'
                try:
                    result = future.result()
                    if (int(result['trial_id']) != tid or result.get('phase') != 'doe'
                            or self._point_key(result) != self._point_key(point)
                            or result.get('status') != 'complete'):
                        raise ValueError('worker returned mismatched DOE result')
                except GeometryDiagnostic as exc:
                    outcome = 'rejected'
                    self._append_csv('diagnostics.csv', DIAGNOSTIC_FIELDS, dict(
                        trial_id=tid, phase='doe', status='geometry_unmeasurable', reason=str(exc),
                        parameters_json=json.dumps(point), workspace=str(self.run_root/f'trial_{tid:04d}')))
                except MeasurementRejected:
                    outcome = 'rejected'
                except Exception as exc:
                    outcome = 'failed'
                    stop_dispatch = True  # Fail closed on technical errors; drain in-flight cases.
                    self._record_status(trial_id=tid, phase='doe', status='failed', stage='failed', message=str(exc))
                self._append_doe_event(dict(event=outcome, trial_id=tid))
                records[tid]['terminal'] = 'complete' if outcome == 'completed' else outcome
                active.pop(tid, None)

        def score():
            if self.config.get('doe_attempt_multiplier') == 5:
                from optimizer.adaptive_refinement import score_rows
                return score_rows(self, self._load_results_csv())
            # execute_trial appends from worker threads. Never rewrite results.csv
            # while another worker appends a completed measurement.
            with self._csv_lock:
                return self.score_completed_results()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                while True:
                    ready = {f for f in pending if f.done()}
                    if ready:
                        ingest(ready)
                        scored = score()
                        continue  # Consume every completed result before choosing.
                    initial_active = any(index is not None for _, _, index in pending.values())
                    initial_finished = not initial_queue and not initial_active
                    capped = len(attempts) >= cap
                    finite = self.config.get('doe_attempt_multiplier') == 5
                    valid_count = sum(r.get('phase') == 'doe' and self._valid_responses(r) for r in scored)
                    target = int(self.config.get('minimum_doe_complete', self.config['campaign']['doe_trials']))
                    if finite:
                        # Reserve at most the missing valid points; wait for feedback before refill.
                        capped = capped or len(records) >= int(self.config['campaign']['max_trials'])
                        if valid_count >= target:
                            stop_dispatch = True
                    elif initial_finished and not (capped and pending):
                        stop_dispatch = True
                    if not pending and (stop_dispatch or capped):
                        break
                    if not stop_dispatch and shutil.disk_usage(self.run_root).free < 15 * 1024**3:
                        stop_dispatch = True  # Preserve disk headroom; finish active cases only.
                    if not stop_dispatch and not capped and len(pending) < workers and (not finite or valid_count + len(pending) < target):
                        proposal_revision = None
                        if initial_queue:
                            initial_index, point = initial_queue[0]
                        elif not initial_active and finite:
                            # A per-ID deterministic stream survives restarts and excludes all history.
                            initial_index = None
                            rng = random.Random(int(self.config.get('random_seed', 1)) * 1000003 + self._next_trial_id)
                            for _ in range(10000):
                                point = {n: float(lo) + rng.random() * (float(hi)-float(lo))
                                         for n, (lo, hi) in bounds.items()}
                                if self._point_key(point) not in observed:
                                    break
                            else:
                                raise RuntimeError('could not generate distinct deterministic DOE backfill')
                        elif not initial_active:
                            from optimizer.directed_doe_adapter import select_for_controller
                            initial_index = None
                            selection_seed += 1
                            seed = selection_seed
                            running_points = [p for _, p, _ in pending.values()]
                            candidates = select_for_controller(self, original, 1, seed, pending_points=running_points)
                            if any(f.done() for f in pending):
                                # Data arrived while fitting; discard and select
                                # again after ingestion. Each audit has its own ID.
                                continue
                            if len(candidates) != 1:
                                raise RuntimeError('informative selector must return exactly one DOE point')
                            point = candidates[0]
                            proposal_revision = getattr(candidates, 'observation_revision', getattr(self, '_results_revision', 0))
                        else:
                            point = None  # Initial barrier, but no whole-batch barrier.
                        if point is not None:
                            if (set(point) != set(self.parameter_names) or not self._is_within_active_bounds(point)
                                    or self._point_key(point) in observed):
                                raise ValueError('invalid, out-of-bounds or duplicate DOE proposal')
                            tid = self._next_trial_id
                            seeds = self._trial_seeds(tid)
                            # Reservation is authoritative and durable BEFORE submit.
                            event=dict(event='reserved', trial_id=tid, parameters=point, seeds=seeds,
                                initial_index=initial_index, execution_order=execution_order if initial_index is not None else 'adaptive_information',
                                pending_trial_ids=sorted(v[0] for v in pending.values()))
                            if not self._commit_doe_reservation(event, proposal_revision, pending):
                                continue
                            attempts.add(tid)
                            observed.add(self._point_key(point))
                            records[tid] = dict(phase='doe', parameters=point, terminal=None)
                            self._next_trial_id += 1
                            if initial_index is not None:
                                initial_queue.pop(0)
                            self._record_status(trial_id=tid, phase='doe', status='reserved', stage='queue',
                                message='fsynced DOE reservation', workspace=str(self.run_root/f'trial_{tid:04d}'))
                            active[tid] = dict(phase='doe', status='preparing', started_at=time.time())
                            pending[executor.submit(self.execute_trial, tid, point, 'doe', seeds)] = (tid, point, initial_index)
                            continue
                    if pending:
                        refresh_live_state(self.run_root, active)
                        write_live_status_csv(self.run_root, active)
                        ready, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                        if ready:
                            ingest(ready)
                            scored = score()
                    else:
                        break
            except BaseException:
                # On stop/error drain only this invocation's known futures and
                # persist terminal outcomes. Never touch unrelated/live solvers.
                if pending:
                    wait(pending)
                    ingest(set(pending))
                raise
            finally:
                write_live_status_csv(self.run_root, active)
        scored = score()
        # Initial campaign ends here. Scientific quality and refinement require
        # a separate decision; no adaptive replacements or calibration run.
        return scored

    def run_doe(self) -> list[dict]:
        if not self.config.get('maximum_doe_attempts'):
            return self._run_doe_legacy()
        import fcntl
        # Same protection for fresh and resume callers; cannot run two schedulers.
        with (self.run_root/'.doe_scheduler.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                return self._run_doe_async()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _run_doe_legacy(self) -> list[dict]:
        scored = self.score_completed_results()
        if self._doe_is_ready(scored, write_report=True):
            _, _, self._next_trial_id = self._resume_doe_state()
            self._calibrate_cohesion_ceiling()
            return scored
        bounds = {name: (spec["min"], spec["max"]) for name, spec in self.config["parameters"].items()}
        manifest_path = self.run_root / "doe_manifest_original.json"
        if manifest_path.is_file():
            points = json.loads(manifest_path.read_text(encoding="utf-8"))["points"]
        else:
            points = self.initial_doe_points()
        completed, rejected_counts, next_trial_id = self._resume_doe_state()
        exhaustion_count = int(self.config.get("measurement_retry_limit", 2)) + 1
        pending = [point for point in points if self._is_within_active_bounds(point) and self._point_key(point) not in completed and rejected_counts.get(self._point_key(point), 0) < exhaustion_count]
        needed = sum(1 for point in points if not self._is_within_active_bounds(point) and self._point_key(point) not in completed)
        observed = {self._point_key(point) for point in points} | completed
        replacements = []
        if needed > 0:
            for point in latin_hypercube(max(50, needed * 10), bounds, int(self.config.get("random_seed", 1)) + 104729):
                if self._point_key(point) not in observed:
                    replacements.append(point)
                    observed.add(self._point_key(point))
                if len(replacements) == needed:
                    break
        if len(replacements) != needed:
            raise RuntimeError("could not generate enough bounded DOE replacement points")
        pending.extend(replacements)
        initial_retries = {self._point_key(point): rejected_counts.get(self._point_key(point), 0) - 1 for point in pending if rejected_counts.get(self._point_key(point), 0)}
        self._run_points(pending, "doe", next_trial_id, initial_retries)
        scored = self.score_completed_results()
        while True:
            if self._doe_is_ready(scored, write_report=True):
                self._calibrate_cohesion_ceiling()
                return scored
            valid_doe = [row for row in scored if row.get("phase") == "doe" and self._valid_responses(row)]
            batch_size = self._doe_backfill_batch_size(len(valid_doe))
            if batch_size == 0:
                self._calibrate_cohesion_ceiling()
                return scored
            completed, rejected_counts, next_trial_id = self._resume_doe_state()
            observed = {self._point_key(point) for point in points} | completed | set(rejected_counts)
            backfill = []
            seed = int(self.config.get("random_seed", 1)) + 209759 + next_trial_id
            if self.config.get('directed_doe_backfill', {}).get('enabled'):
                from optimizer.directed_doe_adapter import select_for_controller
                candidates = select_for_controller(self, points, batch_size, seed)
            else:
                candidates = latin_hypercube(max(50, batch_size * 10), bounds, seed)
            for point in candidates:
                if self._point_key(point) not in observed:
                    backfill.append(point)
                    observed.add(self._point_key(point))
                if len(backfill) == batch_size:
                    break
            if len(backfill) != batch_size:
                raise RuntimeError("could not generate enough DOE contingency points")
            manifest = dict(points=backfill, seed=seed, first_trial_id=next_trial_id, phase='doe', maximum_doe_attempts=self.config.get('maximum_doe_attempts'))
            (self.run_root / f'doe_manifest_backfill_{next_trial_id:04d}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            self._run_points(backfill, "doe", next_trial_id)
            scored = self.score_completed_results()

    def run_campaign(self) -> list[dict]:
        if self.config.get('doe_attempt_multiplier') == 5:
            # Admission must precede any scheduler writes, including DOE
            # reservations. The historical fresh-DOE approval cannot authorize
            # the larger finite campaign.
            from optimizer.launch_authorization import require_launch_authorization
            require_launch_authorization(self.project_root, self.config)
            import fcntl
            with (self.run_root/'.finite_campaign.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    doe = self.run_doe()
                    if not self._doe_is_ready(doe, write_report=True):
                        return doe
                    records = self._restore_async_doe_state()
                    nonrefinement = sum(r['phase'] != 'refinement' for r in records.values())
                    budget = min(int(self.config['campaign']['refinement_trials']),
                                 int(self.config['campaign']['max_trials']) - nonrefinement)
                    if budget < 0:
                        raise RuntimeError('persisted attempts exceed campaign capacity')
                    if budget:
                        from optimizer.adaptive_refinement import run_adaptive_refinement
                        run_adaptive_refinement(self, workers=int(self.config.get('parallel_trials', 1)), budget=budget)
                    from optimizer.adaptive_refinement import score_rows
                    return score_rows(self, self._load_results_csv())
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        doe = self.run_doe()
        refinement = self.nominate_refinement_candidates(doe)
        self._run_points(refinement, "refinement", self._next_trial_id)
        scored = self.score_completed_results()
        completed_refinement = {self._point_key(row) for row in scored
                                if row.get('phase') == 'refinement' and self._valid_responses(row)}
        refinement_count = sum(self._point_key(point) in completed_refinement for point in refinement)
        if refinement_count != len(refinement):
            summary = self.write_validation_summary(scored, 0, len(refinement), refinement_count)
            self._write_cohesion_report(summary['status'])
            return scored
        candidates = self.promising_candidates(scored)
        (self.run_root/'promising_candidates.json').write_text(json.dumps(dict(candidates=candidates,
            purpose='candidate/seed rechecks, NOT independent physical or pilot validation'),indent=2),encoding='utf8')
        validation = [{name: float(best[name]) for name in self.parameter_names} for best in candidates[:min(5,int(self.config['campaign']['validation_trials']))]]
        self._run_points(validation, "validation", self._next_trial_id)
        final_results = self.score_completed_results()
        summary = self.write_validation_summary(final_results, len(validation), len(refinement), refinement_count)
        self._write_cohesion_report("campaign_complete" if summary['campaign_complete'] else summary['status'])
        return final_results

    def write_validation_summary(self, final_results, requested, refinement_requested=0, refinement_complete=0):
        validation_rows = [r for r in final_results if r.get('phase') == 'validation' and self._valid_responses(r)]
        accepted_candidates = [r for r in validation_rows if self.responses_accepted(r)]
        refined = refinement_complete == refinement_requested
        complete = refined and bool(requested) and len(validation_rows) == requested
        status = 'refinement_incomplete' if not refined else ('complete' if complete else 'validation_incomplete')
        summary = dict(
            status=status, campaign_complete=complete,
            refinement_trials_complete=refinement_complete, refinement_trials_requested=refinement_requested,
            validation_trials_complete=len(validation_rows), validation_trials_requested=requested,
            accepted=bool(accepted_candidates), any_candidate_passed=bool(accepted_candidates),
            all_candidates_passed=bool(requested) and len(accepted_candidates)==requested,
            accepted_candidate_count=len(accepted_candidates), accepted_candidates=accepted_candidates,
            best_valid_candidate=min(accepted_candidates,key=lambda r:float(r['score'])) if accepted_candidates else None,
            physical_validation_performed=False, validation_kind='independent stochastic seeds, candidate rechecks only')
        (self.run_root/'campaign_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        return summary

    def promising_candidates(self, rows):
        selected=[];seen=set()
        for row in sorted((r for r in rows if self._valid_responses(r) and r.get('score','') != '' and math.isfinite(float(r['score']))),key=lambda r:(float(r['score']),int(r['trial_id']))):
            key=self._point_key(row)
            if key not in seen:
                selected.append(row);seen.add(key)
            if len(selected)==5: break
        if not selected: raise ValueError('no finite complete candidates for validation')
        return selected

    def smoke(self, trials: int = 3) -> list[dict]:
        if trials < 1:
            raise ValueError("smoke trials must be positive")
        bounds = {name: (spec["min"], spec["max"]) for name, spec in self.config["parameters"].items()}
        points = latin_hypercube(trials, bounds, self.config.get("random_seed", 1))
        workers = min(trials, int(self.config.get("parallel_trials", trials)))
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {}
            point_iterator = iter(enumerate(points, start=1))
            for _ in range(workers):
                try:
                    trial_id, point = next(point_iterator)
                except StopIteration:
                    break
                pending[executor.submit(self.execute_trial, trial_id, point)] = trial_id
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception:
                        for queued in pending:
                            queued.cancel()
                        raise
                    try:
                        trial_id, point = next(point_iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(self.execute_trial, trial_id, point)] = trial_id
        return self.score_completed_results()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke-trials", type=int, default=3)
    parser.add_argument("--parallel-trials", type=int, help="override concurrency without modifying the saved config")
    parser.add_argument("--auto-workers", action="store_true", help="use the smaller of configured max workers and CPUs visible to Python/WSL")
    args = parser.parse_args(argv)
    if args.auto_workers and args.parallel_trials is not None:
        parser.error("--auto-workers cannot be combined with --parallel-trials")
    workers_override = args.parallel_trials
    if args.auto_workers:
        configured_workers = int(load_configuration(args.project_root / "configuracao_otimizacao.txt").get("parallel_trials", 1))
        workers_override = min(configured_workers, max(1, os.cpu_count() or 1))
    controller = CalibrationController(args.project_root, args.run_root, parallel_trials_override=workers_override)
    preflight = controller.preflight()
    if args.preflight:
        print(json.dumps(preflight, indent=2))
        return 0
    if args.smoke:
        print(json.dumps(controller.smoke(args.smoke_trials), indent=2))
        return 0
    if args.run:
        results = controller.run_campaign()
        summary = json.loads((controller.run_root / 'campaign_summary.json').read_text(encoding='utf8'))
        print(json.dumps({
            "status": summary['status'],
            "doe_trials": controller.config["campaign"]["doe_trials"],
            "refinement_trials": controller.config["campaign"]["refinement_trials"],
            "validation_trials": controller.config["campaign"]["validation_trials"],
            "completed_results": len(results),
            "cohesion_limits_report": str(controller.run_root / "cohesion_limits.json"),
            "recommended_ceiling_CED11": (controller._boundary_search_report or {}).get("recommended_ceiling_CED11"),
            "lowest_problematic_CED11": (controller._boundary_search_report or {}).get("lowest_problematic_CED11"),
            "highest_problematic_CED11": (controller._boundary_search_report or {}).get("highest_problematic_CED11"),
        }, indent=2))
        return 0 if summary['campaign_complete'] else 1
    parser.error("choose --preflight, --smoke, or --run")


if __name__ == "__main__":
    sys.exit(main())
