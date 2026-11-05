"""Dependency-free SVG evidence plots for per-trial DEM measurements."""
from __future__ import annotations

import html
import math
import os
import re
import statistics
from pathlib import Path
from textwrap import wrap


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="32" y="38" fill="#263449" font-family="monospace" font-size="24" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="32" y="64" fill="#536477" font-family="monospace" font-size="13">{html.escape(subtitle)}</text>',
    ]


def _text(x, y, text, size=12, color="#263449", weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-family="monospace" font-size="{size}" font-weight="{weight}">{html.escape(str(text))}</text>'


def _discharge_sphere_projection(index, radii, point_count, cx, cy, scale, basal=False):
    """Project the existing measurement slab; do not substitute a zero-width plane."""
    if len(radii) != point_count:
        raise ValueError("particle_radii must have one value per plotted particle")
    radius = float(radii[index])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("plotted particle radii must be finite and positive")
    return (f'<circle data-kind="sphere-projection" data-particle-index="{index}" '
            f'data-radius-sim="{radius:.12g}" data-px-per-unit="{scale:.12g}" '
            f'data-basal-reference="{str(basal).lower()}" cx="{cx:.2f}" cy="{cy:.2f}" '
            f'r="{radius*scale:.2f}" fill="#7d8da1" fill-opacity="{.04 if basal else .07}" '
            f'stroke="#7d8da1" stroke-opacity="{.12 if basal else .4}" stroke-width="1"/>')


def write_discharge_svg(path: Path, points, result: dict, trial_label: str, nbr_note: str = "NBR scaling per trial deck", *, particle_radii=None) -> None:
    """Render one or three base-to-top sliding-angle sections."""
    if result.get("sections"):
        return _write_discharge_three_cuts_svg(path, points, result, trial_label, nbr_note, particle_radii=particle_radii)
    width, height = 1440, 1000
    axis = result["horizontal_axis"].upper()
    hi, ni = (0, 2) if axis == "X" else (2, 0)
    center = float(result.get("centerline", result.get("horizontal_center", 0.0)))
    section_center = float(result.get("section_center", 0.0))
    half_width = float(result.get("section_half_width", 0.1))
    section_bounds = result.get("section_normal_bounds_sim")
    if section_bounds is None:
        normal_low, normal_high = section_center - half_width, section_center + half_width
    else:
        normal_low, normal_high = map(float, section_bounds)
    section_indices = [i for i, p in enumerate(points) if normal_low <= float(p[ni]) <= normal_high]
    section = [(float(points[i][hi]), float(points[i][1]), float(points[i][ni])) for i in section_indices]
    if not section:
        raise ValueError("no particles in the plotted central section")
    xmin, xmax = min(p[0] for p in section), max(p[0] for p in section)
    ymin, ymax = min(p[1] for p in section), max(p[1] for p in section)
    padx = max((xmax - xmin) * .06, 1e-9); pady = max((ymax - ymin) * .09, 1e-9)
    xmin -= padx; xmax += padx; ymin -= pady; ymax += pady
    x0, y0, pw, ph = 90, 190, 1260, 660
    scale = min(pw / (xmax - xmin), ph / (ymax - ymin))
    plot_width, plot_height = (xmax - xmin) * scale, (ymax - ymin) * scale
    offset_x, offset_y = (pw - plot_width) / 2, (ph - plot_height) / 2
    X = lambda x: x0 + offset_x + (x - xmin) * scale
    Y = lambda y: y0 + offset_y + plot_height - (y - ymin) * scale
    navy, green, orange = "#263449", "#14966c", "#ef8c24"
    trial_title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f7f9fc"/>',
           f'<text x="80" y="58" fill="{navy}" font-family="monospace" font-size="38" font-weight="bold">{html.escape(trial_title)} - DESLIZAMENTO</text>',
           f'<text x="80" y="104" fill="{navy}" font-family="monospace" font-size="20">CORTE CENTRAL: {"X" if axis == "Z" else "Z"}={section_center:.3f}    {axis} HORIZONTAL    Y VERTICAL</text>',
           f'<text x="80" y="146" fill="{navy}" font-family="monospace" font-size="20">BUSCA POR ALTURA    FAIXAS DY=DIAMETRO MEDIO {float(result["height_bin_size"]):.3f} M</text>',
           f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="{navy}" stroke-width="3"/>']
    for index in range(1, 5):
        gy = y0 + ph * index / 5
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+pw}" y2="{gy:.1f}" stroke="#d8e0e9" stroke-width="1"/>')
    out.append(f'<line x1="{X(center):.2f}" y1="{y0}" x2="{X(center):.2f}" y2="{y0+ph}" stroke="#14816b" stroke-width="3" stroke-dasharray="14 10"/>')
    out.append(f'<text x="{X(center)+8:.1f}" y="{y0+22}" fill="#14816b" font-family="monospace" font-size="17">{axis}=0</text>')
    for particle_index, (x, y, _) in zip(section_indices, section):
        if particle_radii is not None:
            out.append(_discharge_sphere_projection(particle_index, particle_radii, len(points), X(x), Y(y), scale))
        out.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2.2" fill="#7d8da1" fill-opacity=".76"/>')
    for side, color, angle in (("left", "#d93535", result["left_angle_deg"]), ("right", "#276bd2", result["right_angle_deg"])):
        fitpts = result[f"{side}_fit_points"]
        discarded = result.get(f"{side}_discarded_points", [])
        downweighted = result.get(f"{side}_downweighted_points", [])
        for x, y in [*discarded, *downweighted]:
            cx, cy = X(x), Y(y)
            out.append(f'<path d="M {cx-6:.2f} {cy-6:.2f} L {cx+6:.2f} {cy+6:.2f} M {cx-6:.2f} {cy+6:.2f} L {cx+6:.2f} {cy-6:.2f}" stroke="{orange}" stroke-width="3"/>')
        if not fitpts:
            continue
        bx, by = result.get(f"toe_{side}", result.get(f"base_{side}"))
        denominator = sum((x - bx) ** 2 for x, _ in fitpts)
        slope = float(result.get(f"{side}_slope", sum((x - bx) * (y - by) for x, y in fitpts) / denominator if denominator else 0.0))
        end_x = max(fitpts, key=lambda item: item[1])[0]
        end_y = by + slope * (end_x - bx)
        out.append(f'<line x1="{X(bx):.2f}" y1="{Y(by):.2f}" x2="{X(end_x):.2f}" y2="{Y(end_y):.2f}" stroke="{color}" stroke-width="6"/>')
        for x, y in fitpts:
            out.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="5.2" fill="{green}" stroke="#f7f9fc" stroke-width="1.5"/>')
        out.append(f'<circle cx="{X(bx):.2f}" cy="{Y(by):.2f}" r="8" fill="#ffd84d" stroke="#172536" stroke-width="3"/>')
        sign = -1 if side == "left" else 1
        out.append(f'<line x1="{X(bx):.2f}" y1="{Y(by):.2f}" x2="{X(bx)+sign*58:.2f}" y2="{Y(by):.2f}" stroke="#14966c" stroke-width="4"/>')
        label_x = max(x0+15, min(x0+pw-250, X(bx)+sign*150))
        label_y = max(y0+34, min(y0+ph-28, Y(by)-42))
        out.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" fill="{color}" font-family="monospace" font-size="20" font-weight="bold">ALPHA {float(angle):.2f} GRAUS</text>')
        fit_r2 = float(result.get(f"{side}_r2", 0.0))
        downweighted_count = len(result.get(f"{side}_downweighted_points", []))
        candidate_count = int(result.get(f"{side}_candidate_count", len(fitpts)))
        out.append(f'<text x="{label_x:.1f}" y="{label_y+22:.1f}" fill="{navy}" font-family="monospace" font-size="13">R2 {fit_r2:.3f} | PONTOS {len(fitpts)}/{candidate_count} | PESO REDUZIDO {downweighted_count}</text>')
    out.extend([
        f'<text x="80" y="904" fill="{navy}" font-family="monospace" font-size="19">VERMELHO/AZUL: AJUSTE ROBUSTO DE TODOS OS PONTOS DE PERFIL DA BASE AO TOPO</text>',
        f'<text x="80" y="934" fill="{navy}" font-family="monospace" font-size="19">ESQ {float(result["left_angle_deg"]):.2f} GRAUS   DIR {float(result["right_angle_deg"]):.2f} GRAUS   MEDIA {float(result["mean_angle_deg"]):.2f} GRAUS</text>',
        f'<text x="80" y="964" fill="{navy}" font-family="monospace" font-size="17">X LARANJA: PONTOS DE ALTO RESIDUAL, INCLUIDOS COM PESO ROBUSTO MENOR   PONTO AMARELO: BASE   TRACEJADO: EIXO {axis}=0</text>',
        f'<text x="80" y="988" fill="#536477" font-family="monospace" font-size="13">NBR DECK: {html.escape(nbr_note)}</text>',
        '</svg>'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_svg_png(svg_path: Path, png_path: Path) -> None:
    """Rasterize the source SVG at its native size, without a browser fallback."""
    backend = os.environ.get("DEM_RASTERIZER", "cairosvg")
    if backend != "cairosvg":
        raise ValueError(f"Unsupported DEM_RASTERIZER={backend!r}; use cairosvg or unset DEM_RASTERIZER")
    try:
        import cairosvg
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SVG rasterization requires: python -m pip install CairoSVG==2.9.1 Pillow; "
            "also install native libcairo (Ubuntu/WSL: libcairo2). "
            "The source SVG is preserved; no browser fallback is used."
        ) from exc
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    with Image.open(png_path) as image:
        image.load()
        if image.format != 'PNG':
            raise ValueError('SVG renderer did not create PNG')


def _write_discharge_three_cuts_svg(path: Path, points, result: dict, trial_label: str, nbr_note: str, *, particle_radii=None) -> None:
    """Render the confirmed three NBR-scaled physical cuts side by side in one SVG."""
    width, height = 1800, 1000
    navy, green, red, blue, orange = "#263449", "#14966c", "#d93535", "#276bd2", "#ef8c24"
    axis = result["horizontal_axis"].upper()
    hi, ni = (0, 2) if axis == "X" else (2, 0)
    title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    cuts_label = ", ".join(f"X={float(item['section_center_physical']):+.1f} M" for item in result["sections"])
    half_width_physical = float(result.get("section_half_width_physical", result["sections"][0].get("section_half_width_physical", 0.05)))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="48" y="52" fill="{navy}" font-family="monospace" font-size="34" font-weight="bold">{html.escape(title)} - DESCARGA / DESLIZAMENTO</text>',
        f'<text x="48" y="88" fill="{navy}" font-family="monospace" font-size="17">3 CORTES FISICOS: {html.escape(cuts_label)} | MEIA-LARGURA={half_width_physical:.3f} M | CONVERSAO PELA NBRRATIO</text>',
        f'<text x="48" y="118" fill="{navy}" font-family="monospace" font-size="15">{axis} HORIZONTAL | Y VERTICAL | ESCALA X/Y IGUAL | BASE TRANSLUCIDA: REFERENCIA GEOMETRICA; FORA DOS FITS</text>',
    ]
    panel_w, panel_h, gap_x, margin_x, top = 560, 750, 24, 36, 150
    for section_index, section_result in enumerate(result["sections"]):
        px = margin_x + section_index * (panel_w + gap_x)
        section_center = float(section_result["section_center"])
        half_width = float(section_result["section_half_width"])
        section_bounds = section_result.get("section_normal_bounds_sim")
        if section_bounds is None:
            normal_low, normal_high = section_center - half_width, section_center + half_width
        else:
            normal_low, normal_high = map(float, section_bounds)
        section_indices = [i for i, p in enumerate(points) if normal_low <= float(p[ni]) <= normal_high]
        all_cloud = [(float(points[i][hi]), float(points[i][1])) for i in section_indices]
        if not all_cloud:
            raise ValueError(f"no particles in discharge cut centered at {section_center}")
        centerline = float(section_result["centerline"])
        cutoff_values = {
            "left": float(section_result["profile_start_height_left"]),
            "right": float(section_result["profile_start_height_right"]),
        }
        cloud = all_cloud
        hidden_count = sum(
            y < cutoff_values["left" if x < centerline else "right"]
            for x, y in all_cloud
        )
        xmin, xmax = min(x for x, _ in cloud), max(x for x, _ in cloud)
        ymin, ymax = min(y for _, y in cloud), max(y for _, y in cloud)
        padx, pady = max((xmax - xmin) * .08, 1e-9), max((ymax - ymin) * .08, 1e-9)
        xmin, xmax, ymin, ymax = xmin - padx, xmax + padx, ymin - pady, ymax + pady
        left, plot_top, plot_w, plot_h = px + 34, top + 112, panel_w - 68, panel_h - 200
        scale = min(plot_w / (xmax - xmin), plot_h / (ymax - ymin))
        actual_w, actual_h = (xmax - xmin) * scale, (ymax - ymin) * scale
        off_x, off_y = (plot_w - actual_w) / 2.0, (plot_h - actual_h) / 2.0
        X = lambda x: left + off_x + (float(x) - xmin) * scale
        Y = lambda y: plot_top + off_y + actual_h - (float(y) - ymin) * scale
        physical_center = float(section_result["section_center_physical"])
        half_width_physical = float(section_result.get("section_half_width_physical", result.get("section_half_width_physical", 0.05)))
        effective_band_physical = section_result.get(
            "effective_band_physical", (physical_center - half_width_physical, physical_center + half_width_physical)
        )
        clip_label = " | CLIPADA PELA MALHA" if section_result.get("band_clipped", False) else ""
        out.append(f'<rect x="{px}" y="{top}" width="{panel_w}" height="{panel_h}" rx="8" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+14}" y="{top+30}" fill="{navy}" font-family="monospace" font-size="17" font-weight="bold">CORTE X FISICO | X={physical_center:+.3f} M</text>')
        out.append(f'<text x="{px+14}" y="{top+52}" fill="{navy}" font-family="monospace" font-size="12">CENTRO DEM X={section_center:+.3f} | MEIA-LARGURA NOMINAL={half_width_physical:.3f} M</text>')
        out.append(f'<text x="{px+14}" y="{top+72}" fill="{orange if clip_label else navy}" font-family="monospace" font-size="12">BANDA EFETIVA [{float(effective_band_physical[0]):+.3f},{float(effective_band_physical[1]):+.3f}] M{clip_label}</text>')
        out.append(f'<rect x="{left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        for particle_index, (x, y) in zip(section_indices, cloud):
            side = "left" if x < centerline else "right"
            is_basal_reference = y < cutoff_values[side]
            kind = "basal-reference" if is_basal_reference else "particle"
            opacity = ".16" if is_basal_reference else ".62"
            if particle_radii is not None:
                out.append(_discharge_sphere_projection(particle_index, particle_radii, len(points), X(x), Y(y), scale, is_basal_reference))
            out.append(f'<circle data-kind="{kind}" data-particle-y="{y:.9f}" data-cutoff-y="{cutoff_values[side]:.9f}" data-side="{side}" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2.2" fill="#7d8da1" fill-opacity="{opacity}"/>')
        for side, color, is_left in (("left", red, True), ("right", blue, False)):
            cutoff_y = cutoff_values[side]
            side_x = [x for x, _ in all_cloud if (x < centerline) == is_left]
            if side_x:
                endpoint = min(side_x) if is_left else max(side_x)
                out.append(f'<line data-kind="minimum-height-cutoff" data-cutoff-side="{side}" data-cutoff-y="{cutoff_y:.9f}" x1="{X(centerline):.2f}" y1="{Y(cutoff_y):.2f}" x2="{X(endpoint):.2f}" y2="{Y(cutoff_y):.2f}" stroke="{color}" stroke-width="3" stroke-dasharray="8 5"/>')
        out.append(f'<text data-kind="minimum-height-label" x="{px+14}" y="{top+92}" fill="{orange}" font-family="monospace" font-size="10" font-weight="bold">FIT ESQ Y={cutoff_values["left"]:.4f} | DIR Y={cutoff_values["right"]:.4f} M | BASE TRANSLUCIDA={hidden_count} (REF.)</text>')
        for side, color in (("left", red), ("right", blue)):
            fit_points = section_result[f"{side}_fit_points"]
            bx, by = section_result[f"base_{side}"]
            slope = float(section_result[f"{side}_slope"])
            segments = section_result.get(f"{side}_segments") or [{"fit_points": fit_points, "slope": slope, "base": (bx, by), "angle_deg": section_result[f"{side}_angle_deg"], "fit_support": len(fit_points)}]
            for segment_index, segment in enumerate(segments, 1):
                seg_points = segment["fit_points"]
                seg_bx, seg_by = segment["base"]
                seg_slope = float(segment["slope"])
                seg_end_x = max(seg_points, key=lambda item: item[1])[0]
                seg_end_y = seg_by + seg_slope * (seg_end_x - seg_bx)
                out.append(f'<line data-kind="fit-{side}-segment{segment_index}" x1="{X(seg_bx):.2f}" y1="{Y(seg_by):.2f}" x2="{X(seg_end_x):.2f}" y2="{Y(seg_end_y):.2f}" stroke="{color}" stroke-width="5"/>')
            for x, y in fit_points:
                out.append(f'<circle data-kind="selected" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="4.4" fill="{green}" stroke="#ffffff" stroke-width="1"/>')
            for x, y in [*section_result.get(f"{side}_discarded_points", []), *section_result.get(f"{side}_downweighted_points", [])]:
                cx, cy = X(x), Y(y)
                out.append(f'<path data-kind="outlier" d="M {cx-5:.2f} {cy-5:.2f} L {cx+5:.2f} {cy+5:.2f} M {cx-5:.2f} {cy+5:.2f} L {cx+5:.2f} {cy-5:.2f}" stroke="{orange}" stroke-width="2.5"/>')
            out.append(f'<circle cx="{X(bx):.2f}" cy="{Y(by):.2f}" r="7" fill="#ffd84d" stroke="{navy}" stroke-width="2"/>')
            angle = float(section_result[f"{side}_angle_deg"])
            segment_summary = "  ".join(f"SEG{index}: {float(item['angle_deg']):.2f}°/{int(item['fit_support'])}" for index, item in enumerate(segments, 1))
            label_y = top + panel_h - (64 if side == "left" else 43)
            out.append(f'<text data-kind="segment-label" x="{px+14}" y="{label_y}" fill="{color}" font-family="monospace" font-size="11">{side.upper()} {angle:.2f}° | {segment_summary} (ANGULO/SUPORTE)</text>')
        out.append(f'<text x="{px+18}" y="{top+panel_h-16}" fill="{navy}" font-family="monospace" font-size="14">MEDIA DO CORTE: {float(section_result["mean_angle_deg"]):.2f} GRAUS</text>')
    section_summary = " | ".join(f"X={float(item['section_center_physical']):+.2f}: {float(item['mean_angle_deg']):.2f}°" for item in result["sections"])
    out.extend([
        f'<text x="48" y="940" fill="{navy}" font-family="monospace" font-size="18" font-weight="bold">MEDIA DOS 3 CORTES: {float(result["mean_angle_deg"]):.2f} GRAUS</text>',
        f'<text x="48" y="968" fill="{navy}" font-family="monospace" font-size="15">{html.escape(section_summary)} | NBR DECK: {html.escape(nbr_note)}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_discharge_failure_svg(
    path: Path, points, trial_label: str, reason: str, *,
    horizontal_axis: str, nbrratio: float, mesh_bounds=None,
    physical_centers=(-0.1, 0.0, 0.1), physical_half_width=0.075,
    particle_radii=None, minimum_height_particle_diameters=2.0,
) -> None:
    """Render the standard three physical cuts when a discharge angle is invalid.

    This preserves the normal per-trial layout and provenance but deliberately
    draws no fitted line or angle scalar when the estimator rejects the profile.
    """
    width, height = 1800, 1000
    navy, orange = "#263449", "#ef8c24"
    horizontal_axis = horizontal_axis.lower()
    hi, ni = (0, 2) if horizontal_axis == "x" else (2, 0)
    normal_axis = "z" if horizontal_axis == "x" else "x"
    ratio = float(nbrratio)
    normal_bounds = (mesh_bounds or {}).get(normal_axis)
    horizontal_bounds = (mesh_bounds or {}).get(horizontal_axis, (0.0, 0.0))
    centerline = (float(horizontal_bounds[0]) + float(horizontal_bounds[1])) / 2.0
    minimum_height_particle_diameters = float(minimum_height_particle_diameters)
    if not math.isfinite(minimum_height_particle_diameters) or minimum_height_particle_diameters < 0:
        raise ValueError("minimum_height_particle_diameters must be finite and nonnegative")
    title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    cuts_label = ", ".join(f"X={float(center):+.1f} M" for center in physical_centers)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="48" y="52" fill="{navy}" font-family="monospace" font-size="34" font-weight="bold">{html.escape(title)} - DESCARGA / DESLIZAMENTO</text>',
        f'<text x="48" y="88" fill="{orange}" font-family="monospace" font-size="20" font-weight="bold">NO VALID ANGLE — {html.escape(reason)}</text>',
        f'<text x="48" y="118" fill="{navy}" font-family="monospace" font-size="15">3 CORTES FISICOS: {html.escape(cuts_label)} | MEIA-LARGURA={float(physical_half_width):.3f} M | SEM AJUSTE / SEM ESCALAR</text>',
        f'<text x="48" y="140" fill="{navy}" font-family="monospace" font-size="14">{horizontal_axis.upper()} HORIZONTAL | Y VERTICAL | ESCALA X/Y IGUAL | 2 DIAMETROS ACIMA DA BASE: LIMITE INFERIOR DO AJUSTE (QUANDO RAIO DISPONIVEL)</text>',
    ]
    panel_w, panel_h, gap_x, margin_x, top = 560, 750, 24, 36, 160
    for index, physical_center in enumerate(physical_centers):
        px = margin_x + index * (panel_w + gap_x)
        center = float(physical_center) * ratio
        half = float(physical_half_width) * ratio
        requested = (center - half, center + half)
        effective = requested if normal_bounds is None else (max(requested[0], float(normal_bounds[0])), min(requested[1], float(normal_bounds[1])))
        section_indices = [
            point_index for point_index, point in enumerate(points)
            if effective[0] <= float(point[ni]) <= effective[1]
        ]
        all_cloud = [(float(points[i][hi]), float(points[i][1])) for i in section_indices]
        section_radii = (
            [float(particle_radii[i]) for i in section_indices]
            if particle_radii is not None and len(particle_radii) == len(points) else []
        )
        cutoff_values = {}
        hidden_count = 0
        if section_radii and len(section_radii) == len(all_cloud):
            median_radius = statistics.median(section_radii)
            for side, is_left in (("left", True), ("right", False)):
                side_points = [(x, y) for x, y in all_cloud if (x < centerline) == is_left]
                if side_points:
                    cutoff_values[side] = min(y for _, y in side_points) + minimum_height_particle_diameters * 2.0 * median_radius
            cloud = [(x, y) for x, y in all_cloud if y >= cutoff_values["left" if x < centerline else "right"]]
            hidden_count = len(all_cloud) - len(cloud)
        else:
            cloud = all_cloud
        out.append(f'<rect x="{px}" y="{top}" width="{panel_w}" height="{panel_h}" rx="8" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+14}" y="{top+30}" fill="{navy}" font-family="monospace" font-size="17" font-weight="bold">CORTE X FISICO | X={float(physical_center):+.3f} M</text>')
        clipped = normal_bounds is not None and (effective != requested)
        out.append(f'<text x="{px+14}" y="{top+52}" fill="{orange if clipped else navy}" font-family="monospace" font-size="12">FAIXA DEM [{effective[0]:+.3f},{effective[1]:+.3f}] | NBRRATIO={ratio:.4g}{" | CLIPADA PELA MALHA" if clipped else ""}</text>')
        if cutoff_values:
            label = " | ".join(f"{side.upper()} Y={value:.4f}" for side, value in cutoff_values.items())
            out.append(f'<text data-kind="minimum-height-label" x="{px+14}" y="{top+72}" fill="{orange}" font-family="monospace" font-size="11" font-weight="bold">CORTE MINIMO: {label} M | OCULTAS={hidden_count}</text>')
        left, plot_top, plot_w, plot_h = px+34, top+82, panel_w-68, panel_h-170
        if not cloud:
            out.append(f'<text x="{px+25}" y="{top+panel_h/2}" fill="{orange}" font-family="monospace" font-size="16">SEM PARTICULAS NESTE CORTE</text>')
            continue
        xmin, xmax = min(x for x,_ in cloud), max(x for x,_ in cloud)
        ymin, ymax = min(y for _,y in cloud), max(y for _,y in cloud)
        padx, pady = max((xmax-xmin)*.08, 1e-9), max((ymax-ymin)*.08, 1e-9)
        xmin,xmax,ymin,ymax = xmin-padx,xmax+padx,ymin-pady,ymax+pady
        scale=min(plot_w/(xmax-xmin),plot_h/(ymax-ymin))
        actual_w,actual_h=(xmax-xmin)*scale,(ymax-ymin)*scale
        off_x,off_y=(plot_w-actual_w)/2,(plot_h-actual_h)/2
        X=lambda x:left+off_x+(float(x)-xmin)*scale
        Y=lambda y:plot_top+off_y+actual_h-(float(y)-ymin)*scale
        out.append(f'<rect x="{left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        visible_indices = [i for i in section_indices if not cutoff_values or float(points[i][1]) >= cutoff_values["left" if float(points[i][hi]) < centerline else "right"]]
        for particle_index, (x, y) in zip(visible_indices, cloud):
            if particle_radii is not None:
                out.append(_discharge_sphere_projection(particle_index, particle_radii, len(points), X(x), Y(y), scale))
            out.append(f'<circle data-kind="particle" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2.2" fill="#7d8da1" fill-opacity=".68"/>')
        if cutoff_values:
            for side, color, is_left in (("left", "#d93535", True), ("right", "#276bd2", False)):
                cutoff_y = cutoff_values.get(side)
                side_points = [(x, y) for x, y in all_cloud if (x < centerline) == is_left]
                if cutoff_y is None or not side_points:
                    continue
                endpoint = min(x for x, _ in side_points) if is_left else max(x for x, _ in side_points)
                out.append(f'<line data-kind="minimum-height-cutoff" data-cutoff-side="{side}" data-cutoff-y="{cutoff_y:.9f}" x1="{X(centerline):.2f}" y1="{Y(cutoff_y):.2f}" x2="{X(endpoint):.2f}" y2="{Y(cutoff_y):.2f}" stroke="{color}" stroke-width="3" stroke-dasharray="8 5"/>')
        else:
            out.append(f'<text data-kind="minimum-height-unavailable" x="{px+18}" y="{top+panel_h-58}" fill="{orange}" font-family="monospace" font-size="12" font-weight="bold">CORTE MINIMO INDISPONIVEL: RAIO NAO LIDO</text>')
        out.append(f'<text x="{px+18}" y="{top+panel_h-38}" fill="{orange}" font-family="monospace" font-size="15" font-weight="bold">NO VALID ANGLE / NO FIT</text>')
        out.append(f'<text x="{px+18}" y="{top+panel_h-16}" fill="{navy}" font-family="monospace" font-size="13">{len(cloud)} ABOVE CUTOFF / {hidden_count} HIDDEN</text>')
    out.extend([
        f'<text x="48" y="940" fill="{orange}" font-family="monospace" font-size="20" font-weight="bold">MEDICAO REJEITADA — SEM ANGULO VALIDO; NAO USAR COMO RESULTADO DOE</text>',
        f'<text x="48" y="970" fill="{navy}" font-family="monospace" font-size="14">MOTIVO: {html.escape(reason)} | CENTROS FISICOS: {html.escape(cuts_label)} | NBRRATIO={ratio:.8g}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_repose_svg(path: Path, result: dict, trial_label: str, nbr_note: str = "NBR scaling per trial deck", points=None, *, historical_preview=False) -> None:
    """Render paired radial repose faces, or the legacy eight-ray diagnostic."""
    if result.get("pair_profiles"):
        return _write_repose_paired_svg(path, result, trial_label, nbr_note, points=points, historical_preview=historical_preview)
    width, height = 1440, 1000
    navy, green, blue, orange = "#263449", "#14966c", "#276bd2", "#ef8c24"
    trial_title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="72" y="54" fill="{navy}" font-family="monospace" font-size="36" font-weight="bold">{html.escape(trial_title)} - REPOUSO</text>',
        f'<text x="72" y="88" fill="{navy}" font-family="monospace" font-size="17">8 PERFIS RADIAIS   RAIO MEDIDO NA HORIZONTAL   Y VERTICAL</text>',
        f'<text x="72" y="118" fill="{navy}" font-family="monospace" font-size="15">ANGULO MEDIDO A PARTIR DA HORIZONTAL: atan(dY/dR)   CORRECAO DE RAIO INDIVIDUAL</text>',
    ]
    profiles = result["direction_profiles"]
    cols, rows = 4, 2
    margin_x, margin_y = 72, 150
    gap_x, gap_y = 18, 24
    cw = (width - 2 * margin_x - 3 * gap_x) / 4
    ch = 350
    colors = ["#d93535", "#276bd2"] * 4
    for idx, profile in enumerate(profiles):
        col, row = idx % cols, idx // cols
        px, py = margin_x + col * (cw + gap_x), margin_y + row * (ch + gap_y)
        data = profile["envelope"]
        if not data:
            continue
        rmin = min(r for r, _ in data)
        rmax = max(r for r, _ in data)
        slope, intercept = float(profile["slope"]), float(profile["intercept"])
        center_data = profile.get("center_envelope", [])
        ymin = min([y for _, y in data] + [y for _, y in center_data] + [intercept, slope * rmax + intercept])
        ymax = max([y for _, y in data] + [y for _, y in center_data] + [intercept, slope * rmax + intercept])
        pad_y = max((ymax - ymin) * 0.16, 1e-9)
        ymin, ymax = ymin - pad_y, ymax + pad_y
        rmax = max(rmax, 1e-9)
        left, top, pw, ph = px + 42, py + 45, cw - 58, ch - 100
        scale = min(pw / rmax, ph / (ymax - ymin))
        plot_width, plot_height = rmax * scale, (ymax - ymin) * scale
        offset_x, offset_y = (pw - plot_width) / 2, (ph - plot_height) / 2
        def X(r): return left + offset_x + float(r) * scale
        def Y(y): return top + offset_y + plot_height - (float(y) - ymin) * scale
        out.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="6" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+14:.1f}" y="{py+28:.1f}" fill="{colors[idx%len(colors)]}" font-family="monospace" font-size="18" font-weight="bold">RAIO {idx+1}: {float(profile["angle_deg"]):.2f} GRAUS</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        for grid_index in range(1, 4):
            gy = top + ph * grid_index / 4
            out.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{left+pw}" y2="{gy:.2f}" stroke="#d8e0e9" stroke-width="1"/>')
        # Explicit zero-angle reference: horizontal at the fit's radial-axis intercept.
        ref_y = Y(intercept)
        out.append(f'<line x1="{left}" y1="{ref_y:.2f}" x2="{left+pw}" y2="{ref_y:.2f}" stroke="{green}" stroke-width="2.5" stroke-dasharray="10 7"/>')
        fit_color = colors[idx % len(colors)]
        fit_start, fit_end = min(r for r, _ in data), max(r for r, _ in data)
        out.append(f'<line x1="{X(fit_start):.2f}" y1="{Y(slope*fit_start+intercept):.2f}" x2="{X(fit_end):.2f}" y2="{Y(slope*fit_end+intercept):.2f}" stroke="{fit_color}" stroke-width="4"/>')
        for radius, y in profile.get("center_envelope", []):
            out.append(f'<circle cx="{X(radius):.2f}" cy="{Y(y):.2f}" r="3.0" fill="#7d8da1" fill-opacity=".78"/>')
        for radius, y in data:
            out.append(f'<circle cx="{X(radius):.2f}" cy="{Y(y):.2f}" r="4.0" fill="#14966c" stroke="#ffffff" stroke-width="1"/>')
        out.append(f'<text x="{left}" y="{top+ph+21}" fill="{navy}" font-family="monospace" font-size="12">RAIO R (m)</text>')
        out.append(f'<text x="{left+pw-160}" y="{top+ph+21}" fill="{green}" font-family="monospace" font-size="11">REFERENCIA HORIZONTAL</text>')
        out.append(f'<text x="{left}" y="{top+ph+39}" fill="{navy}" font-family="monospace" font-size="11">{len(data)} PONTOS DE PERFIL | LINHA COLORIDA: AJUSTE</text>')
    mean = float(result["mean_angle_deg"])
    stddev = float(result.get("stddev_deg", 0.0))
    out.extend([
        f'<text x="72" y="922" fill="{navy}" font-family="monospace" font-size="19" font-weight="bold">MEDIA: {mean:.2f} GRAUS   DESVIO ENTRE DIRECOES: {stddev:.2f} GRAUS</text>',
        f'<text x="72" y="952" fill="{navy}" font-family="monospace" font-size="16">ANGULOS: {", ".join(f"{float(a):.2f}" for a in result["direction_angles_deg"])} GRAUS</text>',
        f'<text x="72" y="978" fill="#536477" font-family="monospace" font-size="13">CINZA: CENTROS   VERDE: PERFIL CORRIGIDO   TRACEJADO: REFERENCIA HORIZONTAL (0 GRAUS)   NBR DECK: {html.escape(nbr_note)}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def _write_repose_paired_svg(path: Path, result: dict, trial_label: str, nbr_note: str, points=None, *, historical_preview=False) -> None:
    """Render four diametral views, each with both measured faces and the full particle cloud."""
    width, height = 1800, 1160
    navy, green, red, blue, orange = "#263449", "#14966c", "#d93535", "#276bd2", "#ef8c24"
    cloud = points if points is not None else result.get("particle_cloud", [])
    if not cloud:
        raise ValueError("the paired repose figure requires the saved particle cloud")
    center_x = float(result.get("center_x", 0.0))
    center_z = float(result.get("center_z", 0.0))
    pairs = result.get("pair_profiles", [])
    if len(pairs) != 4:
        raise ValueError("the paired repose figure requires four diametral profiles")
    title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    candidate_method = result.get("profile_method") == "repose_horizontal_surface_consensus"
    valid_faces = sum(profile.get("valid", True) for pair in pairs for profile in (pair["positive_profile"], pair["negative_profile"]))
    summary_label = f"{valid_faces}/8 FACES COM AJUSTE VALIDO" if candidate_method else "8 FACES MEDIDAS INDIVIDUALMENTE"
    axes_label = "EIXOS EXTERNOS DA SECAO" if candidate_method else "EIXOS VERTICAIS NAS BORDAS DAS BASES"
    sampling_label = (f'PASSO EM Y: {float(result.get("height_band_factor",1.0)):.2f}xRAIO 3D MEDIANO' if candidate_method else f'1 PONTO/BANDA: {float(result.get("height_band_factor",2.0)):.2f}×RAIO MEDIANO')
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="48" y="46" fill="{navy}" font-family="monospace" font-size="32" font-weight="bold">{html.escape(title)} - REPOUSO</text>',
        f'<text x="48" y="80" fill="{navy}" font-family="monospace" font-size="17">4 CORTES DIAMETRAIS | {summary_label} | ANGULO DE CADA FACE RELATIVO A Y=0</text>',
        f'<text x="48" y="108" fill="{navy}" font-family="monospace" font-size="15">EIXO CENTRAL (0,0) | {axes_label} | {sampling_label} | CORTE HORIZONTAL: {float(result.get("minimum_height_particle_diameters", 0.0)):.1f} DIAMETROS; AJUSTE COMECA ACIMA DO CORTE</text>',
    ]
    margin_x, margin_y = 48, 132
    panel_w, panel_h, gap_x, gap_y = 840, 432, 24, 22
    colors = ((red, blue),) * 4
    for pair_index, pair in enumerate(pairs):
        col, row = pair_index % 2, pair_index // 2
        px, py = margin_x + col * (panel_w + gap_x), margin_y + row * (panel_h + gap_y)
        theta = math.radians(float(pair["axis_angle_deg"]))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        profile_cutoffs = {
            "right": float(pair["positive_profile"].get("profile_start_height", 0.0)),
            "left": float(pair["negative_profile"].get("profile_start_height", 0.0)),
        }
        # Keep the sub-cutoff bed visible but de-emphasized: it establishes the
        # basal reference/cutoff, while only the opaque profile points enter fits.
        projected_cloud = pair.get("section_projected_centers") or [
            ((float(x) - center_x) * cos_t + (float(z) - center_z) * sin_t, float(y))
            for x, y, z in cloud
        ]
        reference_profiles = [pair["positive_profile"], pair["negative_profile"]]
        base_y_values = [float(profile["base"][1]) for profile in reference_profiles if profile.get("base")]
        base_horizontal_values = [float(profile["base"][0]) for profile in reference_profiles if profile.get("base")]
        base_horizontal_values.extend(
            float(profile["base_axis_coordinate"])
            for profile in reference_profiles if profile.get("base_axis_coordinate") is not None
        )
        max_radius = max([abs(x) for x, _ in projected_cloud] + [abs(x) for x in base_horizontal_values] + [1e-9])
        y_max = max((y for _, y in projected_cloud), default=max(profile_cutoffs.values()) + 1.0)
        xmin, xmax = -max_radius * 1.08, max_radius * 1.08
        ymin = min(list(profile_cutoffs.values()) + base_y_values)
        ymax = max(y_max * 1.06, max(profile_cutoffs.values()) + 1e-9)
        if ymax <= ymin:
            ymax = ymin + 1.0
        left, top, plot_w, plot_h = px + 44, py + 74, panel_w - 86, panel_h - 138
        scale = min(plot_w / (xmax - xmin), plot_h / (ymax - ymin))
        actual_w, actual_h = (xmax - xmin) * scale, (ymax - ymin) * scale
        offset_x, offset_y = (plot_w - actual_w) / 2.0, (plot_h - actual_h) / 2.0
        X = lambda x: left + offset_x + (float(x) - xmin) * scale
        Y = lambda y: top + offset_y + actual_h - (float(y) - ymin) * scale
        color_pos, color_neg = colors[pair_index]
        out.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="8" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+16}" y="{py+30}" fill="{navy}" font-family="monospace" font-size="17" font-weight="bold">CORTE DIAMETRAL {pair_index+1} - EIXO {float(pair["axis_angle_deg"]):.0f} GRAUS</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        out.append(f'<defs><clipPath id="repose-plot-clip-{pair_index}"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>')
        section_radii = pair.get("section_particle_radii", [0.0] * len(projected_cloud))
        for (x, y), sphere_radius in zip(projected_cloud, section_radii):
            side = "right" if x >= 0 else "left"
            is_basal_reference = y < profile_cutoffs[side]
            kind = "basal-reference" if is_basal_reference else "particle"
            opacity = ".16" if is_basal_reference else ".42"
            if sphere_radius:
                out.append(f'<circle data-kind="sphere-section" clip-path="url(#repose-plot-clip-{pair_index})" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="{sphere_radius*scale:.2f}" fill="#7d8da1" fill-opacity="{.04 if is_basal_reference else .07}" stroke="#7d8da1" stroke-opacity="{.12 if is_basal_reference else .4}" stroke-width="1"/>')
            out.append(f'<circle data-kind="{kind}" data-particle-y="{y:.9f}" data-cutoff-y="{profile_cutoffs[side]:.9f}" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="1.7" fill="#7d8da1" fill-opacity="{opacity}"/>')
        out.append(f'<line x1="{X(0):.2f}" y1="{Y(ymin):.2f}" x2="{X(0):.2f}" y2="{Y(ymax):.2f}" stroke="#14816b" stroke-width="2.5" stroke-dasharray="10 7"/>')
        out.append(f'<line x1="{X(xmin):.2f}" y1="{Y(0):.2f}" x2="{X(xmax):.2f}" y2="{Y(0):.2f}" stroke="#14966c" stroke-width="3" stroke-dasharray="10 7"/>')
        out.append(f'<line data-kind="minimum-height-cutoff" data-cutoff-side="left" data-cutoff-y="{profile_cutoffs["left"]:.9f}" x1="{X(xmin):.2f}" y1="{Y(profile_cutoffs["left"]):.2f}" x2="{X(0):.2f}" y2="{Y(profile_cutoffs["left"]):.2f}" stroke="#b42318" stroke-width="3" stroke-dasharray="8 5"/>')
        out.append(f'<line data-kind="minimum-height-cutoff" data-cutoff-side="right" data-cutoff-y="{profile_cutoffs["right"]:.9f}" x1="{X(0):.2f}" y1="{Y(profile_cutoffs["right"]):.2f}" x2="{X(xmax):.2f}" y2="{Y(profile_cutoffs["right"]):.2f}" stroke="#b42318" stroke-width="3" stroke-dasharray="8 5"/>')
        out.append(f'<text x="{left+8}" y="{top+18}" fill="#b42318" font-family="monospace" font-size="12" font-weight="bold">CORTE MINIMO: ESQ Y={profile_cutoffs["left"]:.4f} m | DIR Y={profile_cutoffs["right"]:.4f} m</text>')
        out.append(f'<circle cx="{X(0):.2f}" cy="{Y(0):.2f}" r="6" fill="#ffd84d" stroke="{navy}" stroke-width="2"/>')
        for sign, label, color in ((1.0, "+", color_pos), (-1.0, "-", color_neg)):
            profile = pair["positive_profile"] if sign > 0 else pair["negative_profile"]
            face_cutoff = profile_cutoffs["right" if sign > 0 else "left"]
            base_axis = profile.get("base_axis_coordinate")
            if base_axis is None:
                base = profile.get("base")
                base_axis = base[0] if base else sign * max((abs(float(r)) for r, _ in profile.get("envelope", [])), default=0.0)
            base_axis = float(base_axis)
            axis_base_y = float(profile.get("base", (base_axis, 0.0))[1])
            out.append(f'<line data-kind="toe-axis" x1="{X(base_axis):.2f}" y1="{Y(axis_base_y):.2f}" x2="{X(base_axis):.2f}" y2="{Y(ymax):.2f}" stroke="{orange}" stroke-width="2.5" stroke-dasharray="6 5"/>')
            for x, y, z in profile.get("selected_surface_points", profile.get("selected_particle_points", [])):
                if float(y) < face_cutoff:
                    continue
                projection = (float(x) - center_x) * cos_t + (float(z) - center_z) * sin_t
                out.append(f'<circle data-kind="selected" data-particle-y="{float(y):.9f}" data-cutoff-y="{face_cutoff:.9f}" cx="{X(projection):.2f}" cy="{Y(y):.2f}" r="4.0" fill="{green}" stroke="#ffffff" stroke-width="1"/>')
            downweighted = profile.get("downweighted_particle_points", [])
            for x, y, z in downweighted:
                if float(y) < face_cutoff:
                    continue
                projection = (float(x) - center_x) * cos_t + (float(z) - center_z) * sin_t
                cx, cy = X(projection), Y(y)
                out.append(f'<path data-kind="outlier" data-particle-y="{float(y):.9f}" data-cutoff-y="{face_cutoff:.9f}" d="M {cx-5:.2f} {cy-5:.2f} L {cx+5:.2f} {cy+5:.2f} M {cx-5:.2f} {cy+5:.2f} L {cx+5:.2f} {cy-5:.2f}" stroke="{orange}" stroke-width="2.5"/>')
            for x, y, z in profile.get("outlier_points", []):
                if float(y) < face_cutoff:
                    continue
                projection = (float(x) - center_x) * cos_t + (float(z) - center_z) * sin_t
                cx, cy = X(projection), Y(y)
                out.append(f'<path data-kind="outlier" data-particle-y="{float(y):.9f}" data-cutoff-y="{face_cutoff:.9f}" d="M {cx-5:.2f} {cy-5:.2f} L {cx+5:.2f} {cy+5:.2f} M {cx-5:.2f} {cy+5:.2f} L {cx+5:.2f} {cy-5:.2f}" stroke="{orange}" stroke-width="2.5"/>')
            if result.get("profile_method") in {"discharge_base_anchored_huber", "repose_horizontal_surface_consensus"}:
                base_h, base_y = profile["base"]
                out.append(f'<circle data-kind="base" cx="{X(base_h):.2f}" cy="{Y(base_y):.2f}" r="6" fill="#ffd84d" stroke="{navy}" stroke-width="2"/>')
            data = profile.get("envelope", [])
            face_valid = profile.get("valid", True)
            if not face_valid:
                trace_data = profile.get("profile_points", [])
                trace = " ".join(f"{X(h):.2f},{Y(y):.2f}" for h,y in trace_data)
                out.append(f'<polyline data-kind="selected-profile-trace" clip-path="url(#repose-plot-clip-{pair_index})" points="{trace}" fill="none" stroke="{green}" stroke-width="1.5" stroke-dasharray="3 4"/>')
            if data and face_valid:
                rmin, rmax = min(r for r, _ in data), max(r for r, _ in data)
                slope, intercept = float(profile["slope"]), float(profile["intercept"])
                # The fit may only be drawn from the horizontal cutoff upward.
                if abs(slope) <= 1e-12:
                    if intercept < face_cutoff:
                        rmin, rmax = 1.0, 0.0
                else:
                    cutoff_radius = (face_cutoff - intercept) / slope
                    if slope > 0:
                        rmin = max(rmin, cutoff_radius)
                    else:
                        rmax = min(rmax, cutoff_radius)
                if rmax >= rmin:
                    out.append(f'<line data-kind="fit-{label}" data-cutoff-screen-y="{Y(face_cutoff):.2f}" clip-path="url(#repose-plot-clip-{pair_index})" x1="{X(sign*rmin):.2f}" y1="{Y(slope*rmin+intercept):.2f}" x2="{X(sign*rmax):.2f}" y2="{Y(slope*rmax+intercept):.2f}" stroke="{color}" stroke-width="4"/>')
            angle = pair["positive_angle_deg"] if sign > 0 else pair["negative_angle_deg"]
            angle_label = f"{float(angle):.2f} GRAUS" if face_valid else "SEM ANGULO VALIDO"
            label_x = px + 18 if sign > 0 else px + 430
            r2 = float(profile.get("r2", 0.0))
            support = int(profile.get("fit_support", len(data)))
            candidates = int(profile.get("candidate_count", support))
            downweighted_count = int(profile.get("downweighted_count", len(downweighted)))
            out.append(f'<text x="{label_x}" y="{py+50}" fill="{color}" font-family="monospace" font-size="11" font-weight="bold">LADO {label}: {angle_label} | R2 {r2:.2f} | {support}/{candidates} | EXCLUIDOS {downweighted_count}</text>')
            if not face_valid:
                reason = str(profile.get("rejection_reason", "perfil sem suporte"))
                reason = {"surface does not lean inward with height":"FACE NAO RECUA COM A ALTURA", "insufficient unique particle support":"POUCAS PARTICULAS DISTINTAS", "unsupported height gap in profile":"LACUNA DE ALTURA SEM SUPORTE", "sustained unsupported middle profile":"DESVIO CONTINUO: PERFIL NAO LINEAR", "no majority-supported face":"SEM MAIORIA DE PONTOS COERENTES", "R2 below declared 0.90 face-quality gate":"AJUSTE ABAIXO DO LIMITE R2=0,90", "insufficient inlier support":"POUCOS PONTOS COERENTES"}.get(reason,reason)
                out.append(f'<text data-kind="rejection-reason" x="{px+16}" y="{py+panel_h-44+(0 if sign>0 else 16)}" fill="#b42318" font-family="monospace" font-size="11">LADO {label}: {html.escape(reason)}</text>')
        for gx, gy, gz in pair.get("section_discarded_particle_points", result.get("discarded_particle_points", [])):
            projection = (float(gx) - center_x) * cos_t + (float(gz) - center_z) * sin_t
            local_cutoff = profile_cutoffs["right" if projection >= 0 else "left"]
            if float(gy) < local_cutoff:
                continue
            cx, cy = X(projection), Y(gy)
            out.append(f'<path data-kind="global-outlier" data-particle-y="{float(gy):.9f}" data-cutoff-y="{local_cutoff:.9f}" d="M {cx-5:.2f} {cy-5:.2f} L {cx+5:.2f} {cy+5:.2f} M {cx-5:.2f} {cy+5:.2f} L {cx+5:.2f} {cy-5:.2f}" stroke="{orange}" stroke-width="2.5"/>')
        if pair["mean_angle_deg"] is not None:
            out.append(f'<text x="{left}" y="{top+plot_h+23}" fill="{navy}" font-family="monospace" font-size="12">DISTANCIA ASSINADA AO EIXO (DEM) | REFERENCIA HORIZONTAL Y=0</text>')
        pair_label = f'{float(pair["mean_angle_deg"]):.2f} GRAUS' if pair["mean_angle_deg"] is not None else 'SEM MEDIA VALIDA'
        out.append(f'<text x="{px+16}" y="{py+panel_h-12}" fill="{navy}" font-family="monospace" font-size="13">MEDIA DO PAR: {pair_label} | SUPORTE +/−: {len(pair["positive_fit_points"])}/{len(pair["negative_fit_points"])}</text>')
    angle_text = ", ".join(f"{float(angle):.2f}" if angle is not None else "INVALIDO" for angle in result["direction_angles_deg"])
    legend_text = (
        "NUVEM: CINZA | PERFIL: VERDE | BASE TRANSLUCIDA: REFERENCIA, FORA DO FIT | "
        "RESIDUOS ALTOS: INCLUIDOS COM PESO MENOR (LARANJA) | FORA DOMINIO: X LARANJA"
        if result.get("profile_method") in {"discharge_base_anchored_huber", "repose_horizontal_surface_free_intercept_huber"}
        else "NUVEM: CINZA | PERFIL: VERDE | BASE TRANSLUCIDA: REFERENCIA, FORA DO FIT | REJEITADOS: X LARANJA"
    )
    if result.get("profile_method") == "repose_horizontal_surface_consensus":
        legend_text = "CINZA: ESFERAS DA SECAO | VERDE: SUPERFICIE POR ALTURA | X LARANJA: EXCLUIDOS DO FIT | BASE TRANSLUCIDA: FORA DO FIT"
    mean_label = (f'MEDIA GLOBAL: {float(result["mean_angle_deg"]):.2f} GRAUS | DESVIO ENTRE 8 LADOS: {float(result.get("stddev_deg",0.0)):.2f} GRAUS' if result["mean_angle_deg"] is not None else 'SEM MEDIA GLOBAL: EXISTEM FACES SEM ANGULO VALIDO; NAO USAR NO OBJETIVO')
    provenance_label = "PREVIA DE DUMP HISTORICO; REPROCESSAMENTO SOMENTE LEITURA" if historical_preview else "MEDICAO DO TRIAL"
    out.extend([
        f'<text x="48" y="1040" fill="{navy}" font-family="monospace" font-size="18" font-weight="bold">{mean_label}</text>',
        f'<text x="48" y="1068" fill="{navy}" font-family="monospace" font-size="14">8 ANGULOS INDIVIDUAIS: {angle_text} GRAUS</text>',
        f'<text x="48" y="1096" fill="#536477" font-family="monospace" font-size="13">{html.escape(legend_text)}</text>',
        f'<text x="48" y="1122" fill="#536477" font-family="monospace" font-size="13">{provenance_label} | ORIGEM/EIXO: (0,0) | NBR: {html.escape(nbr_note)}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_repose_cutoff_diagnostic_svg(path: Path, points, trial_label: str, minimum_height: float, reason: str, particle_radii=None) -> None:
    """Standard four-cut repose diagnostic: axes and selected profile, never a false fit."""
    width, height = 1800, 1160
    navy, gray, red = "#263449", "#718096", "#b42318"
    title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="48" y="44" fill="{navy}" font-family="monospace" font-size="30" font-weight="bold">{html.escape(title)} - REPOUSO</text>',
        f'<text x="48" y="78" fill="{red}" font-family="monospace" font-size="19" font-weight="bold">DIAGNOSTICO — SEM ANGULO VALIDO | SEM AJUSTE DE FACE</text>',
        f'<text x="48" y="108" fill="{navy}" font-family="monospace" font-size="15">4 CORTES DIAMETRAIS | PARTICULAS ABAIXO DE Y={float(minimum_height):.4f} m OCULTAS | CORTE VERMELHO</text>',
    ]
    panels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    panel_w, panel_h = 840, 432
    angles = (0.0, 45.0, 90.0, 135.0)
    for idx, (col, row) in enumerate(panels):
        px, py = 48 + col * 864, 132 + row * 442
        theta = math.radians(angles[idx])
        c, s = math.cos(theta), math.sin(theta)
        section = []
        radii = particle_radii if particle_radii is not None and len(particle_radii) == len(points) else [0.0] * len(points)
        for (x, y, z), radius in zip(points, radii):
            h = float(x) * c + float(z) * s
            normal = -float(x) * s + float(z) * c
            radius = float(radius)
            if radius > 0 and abs(normal) < radius:
                radius_in_plane = math.sqrt(max(radius * radius - normal * normal, 0.0))
            else:
                radius_in_plane = radius
            section.append((h, float(y), radius_in_plane))
        visible = [(h, y) for h, y, _ in section if y >= minimum_height]
        section_radius = statistics.median([r for _, _, r in section if r > 0]) if any(r > 0 for _, _, r in section) else 0.0
        band_height = 1.5 * section_radius if section_radius > 0 else max((max(y for _, y, _ in section) - min(y for _, y, _ in section)) / 30.0, 1e-6)
        base_axes = []
        selected_profiles = []
        for side_sign in (1, -1):
            side = [(h, y, r) for h, y, r in section if h * side_sign > 0]
            if not side:
                continue
            base_height = min(y for _, y, _ in side)
            basal = [(h, y, r) for h, y, r in side if y < base_height + band_height]
            edge = max(basal, key=lambda item: item[0])[0] if side_sign > 0 else min(basal, key=lambda item: item[0])[0]
            base_axes.append(edge)
            bins = {}
            for h, y, r in side:
                if y < minimum_height:
                    continue
                band = int((y - base_height) // band_height)
                bins.setdefault(band, []).append((h, y, r))
            chosen = [min(group, key=lambda item: (abs(item[0] - edge), abs(item[1] - (base_height + (band + .5) * band_height)))) for band, group in sorted(bins.items())]
            selected_profiles.append((side_sign, chosen))
        xmax = max((abs(x) for x, _ in visible), default=1.0) * 1.08
        ymax = max((y for _, y in visible), default=minimum_height + 1.0) * 1.05
        ymin = float(minimum_height)
        if ymax <= ymin:
            ymax = ymin + 1.0
        left, top, pw, ph = px + 42, py + 58, panel_w - 76, panel_h - 112
        scale = min(pw / (2 * xmax), ph / (ymax - ymin))
        dw, dh = 2 * xmax * scale, (ymax - ymin) * scale
        ox, oy = (pw - dw) / 2, (ph - dh) / 2
        X = lambda x: left + ox + (float(x) + xmax) * scale
        Y = lambda y: top + oy + dh - (float(y) - ymin) * scale
        out.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="8" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+16}" y="{py+31}" fill="{navy}" font-family="monospace" font-size="18" font-weight="bold">CORTE DIAMETRAL {idx+1} — EIXO {angles[idx]:.0f} GRAUS</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        for x, y in visible:
            out.append(f'<circle data-kind="diagnostic-particle" data-particle-y="{y:.9f}" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2.2" fill="{gray}" fill-opacity=".58"/>')
        out.append(f'<line x1="{X(0):.2f}" y1="{Y(ymin):.2f}" x2="{X(0):.2f}" y2="{Y(ymax):.2f}" stroke="#14816b" stroke-width="2" stroke-dasharray="8 6"/>')
        for edge in base_axes:
            out.append(f'<line data-kind="base-edge-axis" x1="{X(edge):.2f}" y1="{Y(ymin):.2f}" x2="{X(edge):.2f}" y2="{Y(ymax):.2f}" stroke="#ef8c24" stroke-width="2.5" stroke-dasharray="7 5"/>')
        for side_idx, (side_sign, chosen) in enumerate(selected_profiles):
            color = ("#d93535", "#276bd2")[side_idx % 2]
            if len(chosen) > 1:
                trace = " ".join(f"{X(h):.2f},{Y(y):.2f}" for h, y, _ in chosen)
                out.append(f'<polyline data-kind="selected-profile-trace" points="{trace}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="5 4"/>')
            for h, y, _ in chosen:
                out.append(f'<circle data-kind="selected-profile-point" data-particle-y="{y:.9f}" cx="{X(h):.2f}" cy="{Y(y):.2f}" r="4.2" fill="{color}" stroke="#fff" stroke-width="1"/>')
        out.append(f'<line data-kind="minimum-height-cutoff" x1="{left}" y1="{Y(minimum_height):.2f}" x2="{left+pw}" y2="{Y(minimum_height):.2f}" stroke="{red}" stroke-width="3" stroke-dasharray="9 5"/>')
        out.append(f'<text x="{left+10}" y="{top+20}" fill="{red}" font-family="monospace" font-size="12" font-weight="bold">CORTE MINIMO: Y={minimum_height:.4f} m</text>')
    out.extend([
        f'<text x="48" y="1040" fill="{navy}" font-family="monospace" font-size="14">RAZAO: {html.escape(str(reason)[:180])}</text>',
        f'<text x="48" y="1070" fill="{navy}" font-family="monospace" font-size="14">CINZA: NUVEM | TRACEJADO VERDE: EIXO CENTRAL | LARANJA: EIXO DA BORDA | VERMELHO/AZUL: PONTOS/TRAÇO SELECIONADOS (SEM AJUSTE) | CORTE VERMELHO</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(out), encoding="utf-8")


def write_repose_failure_diagnostic_svg(path: Path, points, trial_label: str, reason: str, particle_radii=None) -> None:
    """Render a failed repose snapshot in the standard four-cut layout, without inventing a fit."""
    width, height = 1800, 1160
    navy, gray, red = "#263449", "#718096", "#b42318"
    title = re.sub(r"^TRIAL[_-]", "TRIAL ", trial_label.upper())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="48" y="44" fill="{navy}" font-family="monospace" font-size="30" font-weight="bold">{html.escape(title)} - REPOUSO</text>',
        f'<text x="48" y="78" fill="{red}" font-family="monospace" font-size="19" font-weight="bold">DIAGNOSTICO — SEM ANGULO VALIDO | SEM AJUSTE DE FACE</text>',
        f'<text x="48" y="108" fill="{navy}" font-family="monospace" font-size="15">4 CORTES DIAMETRAIS | NUVEM REAL DO SNAPSHOT | SEM LINHA OU VALOR DE ANGULO</text>',
    ]
    angles = (0.0, 45.0, 90.0, 135.0)
    for idx, angle in enumerate(angles):
        px, py = 48 + (idx % 2) * 864, 132 + (idx // 2) * 442
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        section = [(float(x) * c + float(z) * s, float(y)) for index,(x,y,z) in enumerate(points)
                   if particle_radii is None or abs(-float(x)*s+float(z)*c) < float(particle_radii[index])]
        xmin = min([x for x, _ in section] + [0.0]); xmax = max([x for x, _ in section] + [0.0])
        ymin = min([y for _, y in section] + [0.0]); ymax = max([y for _, y in section] + [0.0])
        padx = max((xmax - xmin) * .08, 1e-9); pady = max((ymax - ymin) * .08, 1e-9)
        xmin -= padx; xmax += padx; ymin -= pady; ymax += pady
        panel_w, panel_h = 840, 432
        left, top, pw, ph = px + 42, py + 58, panel_w - 76, panel_h - 112
        scale = min(pw / (xmax - xmin), ph / (ymax - ymin))
        dw, dh = (xmax - xmin) * scale, (ymax - ymin) * scale
        ox, oy = (pw - dw) / 2, (ph - dh) / 2
        X = lambda x: left + ox + (float(x) - xmin) * scale
        Y = lambda y: top + oy + dh - (float(y) - ymin) * scale
        out.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="8" fill="#f8fafc" stroke="#263449" stroke-width="2"/>')
        out.append(f'<text x="{px+16}" y="{py+31}" fill="{navy}" font-family="monospace" font-size="18" font-weight="bold">CORTE DIAMETRAL {idx+1} — EIXO {angle:.0f} GRAUS</text>')
        out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="{navy}" stroke-width="1.5"/>')
        for x, y in section:
            out.append(f'<circle data-kind="particle" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2.2" fill="{gray}" fill-opacity=".58"/>')
        out.append(f'<line data-kind="center-axis" x1="{X(0):.2f}" y1="{Y(ymin):.2f}" x2="{X(0):.2f}" y2="{Y(ymax):.2f}" stroke="#14816b" stroke-width="2" stroke-dasharray="8 6"/>')
        out.append(f'<line data-kind="horizontal-reference" x1="{X(xmin):.2f}" y1="{Y(0):.2f}" x2="{X(xmax):.2f}" y2="{Y(0):.2f}" stroke="#14966c" stroke-width="2" stroke-dasharray="8 6"/>')
    out.extend([
        f'<text x="48" y="1040" fill="{red}" font-family="monospace" font-size="15" font-weight="bold">MOTIVO: {html.escape(str(reason)[:220])}</text>',
        f'<text x="48" y="1070" fill="{navy}" font-family="monospace" font-size="14">CINZA: PARTICULAS DO SNAPSHOT | VERDE: EIXO CENTRAL E REFERENCIA HORIZONTAL</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_measurement_diagnostic_svg(path: Path, points, kind: str, trial_label: str, reason: str, horizontal_axis: str = "z", minimum_height: float | None = None) -> None:
    """Plot the actual saved snapshot without displaying repose particles below its measurement cutoff."""
    reason_lines = wrap(f"Reason: {reason}", width=128, break_on_hyphens=False) or ["Reason:"]
    width, height = 1120, max(720, 704 + 18 * (len(reason_lines) - 1) + 20)
    label = "discharge/sliding" if kind == "discharge" else "repose"
    out = _svg_header(width, height, f"{trial_label} · {label} measurement diagnostic", "Diagnostic projection only — no angle assigned; not a measurement result")
    x0, y0, pw, ph = 95, 120, 900, 500
    if kind == "discharge":
        hi = 2 if horizontal_axis.lower() == "z" else 0
        values = [(float(p[hi]), float(p[1])) for p in points]
        xlabel = f"{horizontal_axis.upper()} (m)"
    else:
        values = [(math.hypot(float(p[0]), float(p[2])), float(p[1])) for p in points]
        xlabel = "radial distance from vertical axis (m)"
    original_count = len(values)
    below_cutoff_count = 0
    if kind == "repose" and minimum_height is not None:
        below_cutoff_count = sum(1 for _, y in values if y < float(minimum_height))
        values = [(x, y) for x, y in values if y >= float(minimum_height)]
    if original_count > 50000:
        stride = math.ceil(original_count / 50000)
        values = values[::stride]
        sampling_note = f"Showing deterministic 1/{stride} sample of {len(values)} centers above cutoff"
    else:
        sampling_note = f"Showing {len(values)} centers above cutoff; {below_cutoff_count} below cutoff hidden" if below_cutoff_count else f"Showing all {len(values)} particle centers"
    out.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#263449"/>')
    if values:
        xs = [p[0] for p in values]
        ys = [p[1] for p in values]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(p[1] for p in values), max(ys)
        if kind == "repose" and minimum_height is not None:
            ymin = float(minimum_height)
        padx = max((xmax - xmin) * .05, 1e-9)
        pady = max((ymax - ymin) * .08, 1e-9)
        xmin -= padx; xmax += padx; ymin -= pady; ymax += pady
        def X(x): return x0 + (x - xmin) / (xmax - xmin) * pw
        def Y(y): return y0 + ph - (y - ymin) / (ymax - ymin) * ph
        for x, y in values:
            out.append(f'<circle data-kind="diagnostic-particle" data-particle-y="{y:.9f}" cx="{X(x):.2f}" cy="{Y(y):.2f}" r="2" fill="#7d8da1" fill-opacity=".62"/>')
        out.append(f'<line x1="{x0}" y1="{Y(0):.2f}" x2="{x0+pw}" y2="{Y(0):.2f}" stroke="#14966c"/>')
        if kind == "repose" and minimum_height is not None:
            out.append(f'<line data-kind="minimum-height-cutoff" x1="{x0}" y1="{Y(minimum_height):.2f}" x2="{x0+pw}" y2="{Y(minimum_height):.2f}" stroke="#d93535" stroke-width="3" stroke-dasharray="10 6"/>')
            out.append(_text(x0 + 12, Y(minimum_height) - 10, f"MINIMUM PROFILE HEIGHT: {float(minimum_height):.4f} m — LOWER POINTS HIDDEN", 13, "#b42318"))
    else:
        out.append(_text(x0 + 30, y0 + 60, "Snapshot contains no plottable particle centers", 16, "#ef8c24"))
    out.extend([
        _text(x0, y0 + ph + 28, xlabel),
        _text(22, y0 + ph / 2, "Y (m)"),
        _text(96, 680, sampling_note, 12, "#536477"),
    ])
    out.extend(_text(96, 704 + 18 * index, line, 12, "#ef8c24").replace(
        '<text ', '<text data-kind="diagnostic-reason" ', 1)
        for index, line in enumerate(reason_lines))
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
