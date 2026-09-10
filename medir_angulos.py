#!/usr/bin/env python3
"""Mede angulo de repouso e angulo de cisalhamento em snapshots VTK do LIGGGHTS."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def read_vtk_points(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="ascii", errors="strict")
    header = re.search(r"POINTS\s+(\d+)\s+\w+\s*\n", text)
    if not header:
        raise ValueError(f"Cabecalho POINTS nao encontrado em {path}")
    count = int(header.group(1))
    tail = text[header.end():]
    end = re.search(r"\n(?:VERTICES|POINT_DATA|CELLS)\b", tail)
    values = [float(value) for value in (tail[:end.start()] if end else tail).split()]
    if len(values) != count * 3:
        raise ValueError(f"VTK incompleto: esperados {count*3} valores, encontrados {len(values)}")
    return list(zip(values[0::3], values[1::3], values[2::3]))


def surface_profile(points: list[tuple[float, float, float]], bins: int = 100) -> list[tuple[float, float]]:
    # Plano de medicao y-z: y vertical; z na direcao de abertura das bases.
    z_values = [point[2] for point in points]
    z_min, z_max = quantile(z_values, 0.002), quantile(z_values, 0.998)
    width = (z_max - z_min) / bins
    result = []
    for index in range(bins):
        low, high = z_min + index * width, z_min + (index + 1) * width
        y_values = [point[1] for point in points if low <= point[2] < high]
        if len(y_values) >= 6:
            # Percentil alto reduz a influencia de vazios sem seguir particulas isoladas.
            result.append(((low + high) / 2.0, quantile(y_values, 0.95)))
    if len(result) < 20:
        raise ValueError("Poucos pontos para reconstruir a superficie livre")
    return result


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 5:
        raise ValueError("Trecho de superficie insuficiente para regressao")
    xs, ys = zip(*points)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "slope": slope,
        "intercept": intercept,
        "angle_deg": math.degrees(math.atan(abs(slope))),
        "r2": 1.0 - residual / total if total else 1.0,
        "points": len(points),
    }


def select_height_band(points, baseline, peak, low_fraction=0.18, high_fraction=0.82):
    height = peak - baseline
    low = baseline + low_fraction * height
    high = baseline + high_fraction * height
    return [(z, y) for z, y in points if low <= y <= high]


def measure_repose(path: Path) -> dict:
    profile = surface_profile(read_vtk_points(path))
    baseline = quantile([y for _, y in profile], 0.05)
    peak_z, peak_y = max(profile, key=lambda item: item[1])
    left = select_height_band([(z, y) for z, y in profile if z < peak_z], baseline, peak_y)
    right = select_height_band([(z, y) for z, y in profile if z > peak_z], baseline, peak_y)
    left_fit, right_fit = linear_fit(left), linear_fit(right)
    mean = statistics.fmean([left_fit["angle_deg"], right_fit["angle_deg"]])
    return {
        "mean_deg": mean,
        "left": left_fit,
        "right": right_fit,
        "asymmetry_deg": abs(left_fit["angle_deg"] - right_fit["angle_deg"]),
        "baseline_y": baseline,
        "peak": {"z": peak_z, "y": peak_y},
    }


def measure_drawdown(path: Path) -> dict:
    profile = surface_profile(read_vtk_points(path))
    baseline = quantile([y for _, y in profile], 0.03)
    left_half = [(z, y) for z, y in profile if z < 0.0]
    right_half = [(z, y) for z, y in profile if z > 0.0]
    left_peak = max(left_half, key=lambda item: item[1])
    right_peak = max(right_half, key=lambda item: item[1])
    # Somente as faces internas: do pico de cada lado em direcao a abertura central.
    left_inner = [(z, y) for z, y in left_half if z > left_peak[0]]
    right_inner = [(z, y) for z, y in right_half if z < right_peak[0]]
    left = select_height_band(left_inner, baseline, left_peak[1])
    right = select_height_band(right_inner, baseline, right_peak[1])
    left_fit, right_fit = linear_fit(left), linear_fit(right)
    mean = statistics.fmean([left_fit["angle_deg"], right_fit["angle_deg"]])
    return {
        "mean_deg": mean,
        "left": left_fit,
        "right": right_fit,
        "asymmetry_deg": abs(left_fit["angle_deg"] - right_fit["angle_deg"]),
        "baseline_y": baseline,
        "left_peak": {"z": left_peak[0], "y": left_peak[1]},
        "right_peak": {"z": right_peak[0], "y": right_peak[1]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repose_vtk", type=Path)
    parser.add_argument("drawdown_vtk", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = {
        "repose": measure_repose(args.repose_vtk),
        "drawdown": measure_drawdown(args.drawdown_vtk),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Repouso:  {result['repose']['mean_deg']:.3f} deg")
        print(f"Drawdown: {result['drawdown']['mean_deg']:.3f} deg")


if __name__ == "__main__":
    main()

