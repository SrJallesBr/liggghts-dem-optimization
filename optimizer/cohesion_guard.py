"""Track measured and unmeasurable cohesion values and recommend a safe next ceiling."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CohesionLimitTracker:
    minimum: float
    maximum: float
    max_probes: int = 8
    valid: list[dict] = field(default_factory=list)
    problematic: list[dict] = field(default_factory=list)
    probes_issued: int = 0

    def _record(self, destination: list[dict], value: float, trial_id: int, parameters: dict, reason: str = "") -> None:
        destination.append({
            "CED11": float(value),
            "trial_id": int(trial_id),
            "parameters": {key: float(item) for key, item in parameters.items()},
            "reason": reason,
        })

    def observe_valid(self, value: float, trial_id: int, parameters: dict) -> None:
        self._record(self.valid, value, trial_id, parameters)

    def observe_problematic(self, value: float, trial_id: int, parameters: dict, reason: str) -> None:
        self._record(self.problematic, value, trial_id, parameters, reason)

    @property
    def lowest_problematic(self) -> float | None:
        values = [row["CED11"] for row in self.problematic]
        return min(values) if values else None

    @property
    def highest_problematic(self) -> float | None:
        values = [row["CED11"] for row in self.problematic]
        return max(values) if values else None

    @property
    def highest_valid_below_problem(self) -> float | None:
        ceiling = self.lowest_problematic
        values = [row["CED11"] for row in self.valid if ceiling is None or row["CED11"] < ceiling]
        return max(values) if values else None

    @property
    def recommended_ceiling(self) -> float:
        known_valid = self.highest_valid_below_problem
        return float(self.maximum if known_valid is None and self.lowest_problematic is None else self.minimum if known_valid is None else known_valid)

    def next_probe(self) -> float | None:
        low = self.highest_valid_below_problem
        high = self.lowest_problematic
        if low is None or high is None or self.probes_issued >= self.max_probes or high <= low:
            return None
        self.probes_issued += 1
        return (low + high) / 2.0

    def report(self) -> dict:
        next_value = None
        low = self.highest_valid_below_problem
        high = self.lowest_problematic
        if low is not None and high is not None and self.probes_issued < self.max_probes and high > low:
            next_value = (low + high) / 2.0
        return {
            "minimum_CED11": float(self.minimum),
            "configured_maximum_CED11": float(self.maximum),
            "highest_measurable_CED11": self.highest_valid_below_problem,
            "lowest_problematic_CED11": self.lowest_problematic,
            "highest_problematic_CED11": self.highest_problematic,
            "recommended_ceiling_CED11": self.recommended_ceiling,
            "next_probe_CED11": next_value,
            "max_probes": int(self.max_probes),
            "probes_issued": int(self.probes_issued),
            "valid_observations": self.valid,
            "problematic_observations": self.problematic,
        }
