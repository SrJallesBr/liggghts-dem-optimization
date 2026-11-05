import math
import re
import statistics


def detect_sliding_onset(rows, speed_threshold, consecutive_samples):
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be at least 1")
    streak = []
    for row in rows:
        if abs(float(row["speed"])) >= speed_threshold:
            streak.append(row)
            if len(streak) == consecutive_samples:
                return streak[0]
        else:
            streak.clear()
    raise ValueError("no sustained sliding onset found")


def parse_sliding_thermo(text):
    rows = []
    previous = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        try:
            step, time, atoms, angle_deg, com_x, com_y, com_z, along_base = map(float, fields)
        except ValueError:
            continue
        if angle_deg < 0 or (previous is not None and time <= previous["time"]):
            continue
        speed = 0.0 if previous is None else (along_base - previous["com_along_base"]) / (time - previous["time"])
        row = {"step": int(step), "time": time, "atoms": int(atoms), "angle_deg": angle_deg, "com_x": com_x, "com_y": com_y, "com_z": com_z, "com_along_base": along_base, "speed": speed}
        rows.append(row)
        previous = row
    if not rows:
        raise ValueError("no sliding thermo rows found")
    return rows


class UnmeasurableAngle(ValueError):
    """The surface geometry does not define a reliable pile-face angle."""


def parse_ascii_stl_bounds(text):
    """Retorna limites XYZ de um STL ASCII, inclusive quando o STL já está escalado pelo NBR ratio."""
    vertices = []
    for line in text.splitlines():
        match = re.match(r"\s*vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$", line, re.IGNORECASE)
        if match:
            vertices.append(tuple(map(float, match.groups())))
    if not vertices:
        raise ValueError("ASCII STL has no vertex records")
    return {axis: (min(point[index] for point in vertices), max(point[index] for point in vertices)) for index, axis in enumerate(("x", "y", "z"))}


def parse_vtk_points(text):
    match = re.search(r"^POINTS\s+(\d+)\s+\w+\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError("VTK POINTS section not found")
    count = int(match.group(1)); remainder = text[match.end():]
    end = re.search(r"^VERTICES\b|^POLYGONS\b|^POINT_DATA\b", remainder, re.MULTILINE)
    values = [float(value) for value in remainder[:end.start() if end else len(remainder)].split()]
    if len(values) < count * 3:
        raise ValueError("VTK POINTS section is truncated")
    return [tuple(values[index:index + 3]) for index in range(0, count * 3, 3)]


def parse_vtk_scalar_field(text, name):
    """Parse one scalar array from an ASCII legacy VTK FIELD section."""
    match = re.search(rf"(?m)^{re.escape(name)}\s+1\s+(\d+)\s+\w+\s*$", text)
    if match is None:
        raise ValueError(f"VTK scalar field {name!r} not found")
    count = int(match.group(1))
    payload = text[match.end():].lstrip("\r\n ")
    payload = re.split(r"\r?\n\s*\r?\n", payload, maxsplit=1)[0]
    tokens = payload.split()
    if len(tokens) < count:
        raise ValueError(f"VTK scalar field {name!r} expected {count} values, found {len(tokens)}")
    try:
        values = [float(value) for value in tokens[:count]]
    except ValueError as exc:
        raise ValueError(f"VTK scalar field {name!r} contains non-numeric data among its declared values") from exc
    return values


def _linear_slope(points):
    mean_z = sum(point[0] for point in points) / len(points); mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((z - mean_z) ** 2 for z, _ in points)
    if denominator == 0:
        raise ValueError("insufficient Z spread for V-side fit")
    return sum((z - mean_z) * (y - mean_y) for z, y in points) / denominator


def detect_vertical_profile_arching(points, particle_radii, *, horizontal_axis, mesh_bounds=None, section_half_width=0.1, max_horizontal_jump_radii=5.0, min_consecutive_segments=2):
    """Detect sustained outward/re-entrant motion in height-binned side profiles.

    Angles are oriented from the local inward-facing vertical direction toward
    the outward radial direction: 90° is vertical, <90° leans inward as height
    rises, and >90° means the observed surface turns outward while rising.
    A diagnosis requires consecutive >90° segments on at least one side.
    """
    if horizontal_axis not in {"x", "z"}:
        raise ValueError("horizontal_axis must be 'x' or 'z' because Y is vertical")
    if len(points) != len(particle_radii):
        raise ValueError("particle_radii must have one value per VTK particle")
    if not points or any(float(radius) <= 0 for radius in particle_radii):
        raise ValueError("arching detection requires positive particle radii")
    if section_half_width <= 0 or max_horizontal_jump_radii <= 0 or min_consecutive_segments < 1:
        raise ValueError("arching detection tolerances must be positive")
    horizontal_index = 0 if horizontal_axis == "x" else 2
    normal_index = 2 if horizontal_axis == "x" else 0
    normal_axis = "z" if horizontal_axis == "x" else "x"
    mesh_bounds = mesh_bounds or {}
    normal_bounds = mesh_bounds.get(normal_axis, (-section_half_width, section_half_width))
    horizontal_bounds = mesh_bounds.get(horizontal_axis, (0.0, 0.0))
    section_center = (float(normal_bounds[0]) + float(normal_bounds[1])) / 2.0
    centerline = (float(horizontal_bounds[0]) + float(horizontal_bounds[1])) / 2.0
    section = [(float(point[horizontal_index]), float(point[1]), float(radius))
               for point, radius in zip(points, particle_radii)
               if abs(float(point[normal_index]) - section_center) <= section_half_width]
    if len(section) < 4:
        raise ValueError("insufficient particles in central section for arching detection")
    median_radius = statistics.median(radius for _, _, radius in section)
    band_height = 2.0 * median_radius
    sides = {}
    for side_name, side_sign in (("left", -1.0), ("right", 1.0)):
        side = [item for item in section if (item[0] - centerline) * side_sign > 0]
        if not side:
            sides[side_name] = {"detected": False, "max_directed_angle_deg": 0.0, "consecutive_overhang_segments": 0, "segments": [], "profile": []}
            continue
        base_y = min(y for _, y, _ in side)
        top_y = max(y for _, y, _ in side)
        band_count = int(math.floor((top_y - base_y) / band_height)) + 1
        profile = []
        for band_index in range(band_count):
            low = base_y + band_index * band_height
            high = low + band_height
            group = [item for item in side if low <= item[1] < high or (band_index == band_count - 1 and item[1] == high)]
            if group:
                x, y, radius = min(group, key=lambda item: (abs(item[0] - centerline), abs(item[1] - (low + high) / 2.0)))
                profile.append({"band": band_index, "x": x, "y": y, "particle_radius": radius, "outward_distance": (x - centerline) * side_sign + radius})
        kept = []
        discarded = []
        for sample in profile:
            if kept and abs(sample["x"] - kept[-1]["x"]) > max_horizontal_jump_radii * median_radius:
                discarded.append(sample)
                continue
            kept.append(sample)
        segments = []
        streak = 0
        max_streak = 0
        for lower, upper in zip(kept, kept[1:]):
            dy = upper["y"] - lower["y"]
            dr = upper["outward_distance"] - lower["outward_distance"]
            contiguous = upper["band"] == lower["band"] + 1
            angle = 90.0 + math.degrees(math.atan2(dr, dy)) if dy > 0 else 180.0
            overhang = contiguous and angle > 90.0
            streak = streak + 1 if overhang else 0
            max_streak = max(max_streak, streak)
            segments.append({"lower_band": lower["band"], "upper_band": upper["band"], "directed_angle_deg": angle, "outward_shift": dr, "overhang": overhang})
        sides[side_name] = {
            "detected": max_streak >= min_consecutive_segments,
            "max_directed_angle_deg": max((segment["directed_angle_deg"] for segment in segments), default=0.0),
            "consecutive_overhang_segments": max_streak,
            "segments": segments,
            "profile": kept,
            "discarded_profile_samples": discarded,
        }
    return {
        "detected": any(side["detected"] for side in sides.values()),
        "criterion": f">90 degrees for at least {min_consecutive_segments} consecutive diameter-height segments",
        "horizontal_axis": horizontal_axis,
        "centerline": centerline,
        "section_center": section_center,
        "height_bin_size": band_height,
        "median_radius": median_radius,
        "sides": sides,
    }


def measure_repose_eight_ray_least_squares_legacy(points, radial_bins=None, min_points_per_ray=8, particle_radii=None):
    """Measure eight radial faces from the fixed (0,0) axis with local outlier screening."""
    if len(points) < 80:
        raise ValueError("insufficient particles for eight-ray repose measurement")
    if particle_radii is None:
        particle_radii = [0.0] * len(points)
    if len(particle_radii) != len(points):
        raise ValueError("particle_radii must have one value per VTK particle")
    particle_radii = [float(radius) for radius in particle_radii]
    if any(radius < 0 for radius in particle_radii):
        raise ValueError("particle radii cannot be negative")

    center_x = center_z = 0.0
    center_radii = [math.hypot(x, z) for x, _, z in points]
    sorted_center_radii = sorted(center_radii)
    radial_cap = 1.5 * sorted_center_radii[round((len(sorted_center_radii) - 1) * 0.75)]
    retained_indices = [i for i, radius in enumerate(center_radii) if radius <= radial_cap]
    discarded_indices = [i for i, radius in enumerate(center_radii) if radius > radial_cap]
    retained = [
        (float(points[i][0]), float(points[i][1]), float(points[i][2]), center_radii[i], particle_radii[i], i)
        for i in retained_indices
    ]
    radial_max = max(item[3] for item in retained)
    if radial_max <= 0:
        raise ValueError("insufficient radial spread for repose measurement")
    radial_bins = radial_bins or max(8, min(16, len(points) // 70))
    if radial_bins < 4:
        raise ValueError("radial_bins must be at least 4")
    median_particle_radius = statistics.median(particle_radii[i] for i in retained_indices)
    half_sector = math.pi / 8.0

    def quantile(values, fraction):
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    direction_angles = []
    supports = []
    direction_profiles = []
    for direction_index in range(8):
        theta = 2.0 * math.pi * direction_index / 8.0
        radial_points = []
        for x, y, z, radial_distance, particle_radius, point_index in retained:
            if radial_distance == 0:
                continue
            point_theta = math.atan2(z, x)
            angular_distance = abs(math.atan2(math.sin(point_theta - theta), math.cos(point_theta - theta)))
            if angular_distance <= half_sector:
                radial_points.append((radial_distance, y, particle_radius, point_index))

        center_envelope = []
        selected_particle_points = []
        outlier_particle_points = []
        for bin_index in range(radial_bins):
            low = radial_max * bin_index / radial_bins
            high = radial_max * (bin_index + 1) / radial_bins
            group = [item for item in radial_points if low <= item[0] < high or (bin_index == radial_bins - 1 and item[0] == high)]
            if not group:
                continue
            heights = [item[1] for item in group]
            median_height = statistics.median(heights)
            mad = statistics.median(abs(value - median_height) for value in heights)
            upper_limit = quantile(heights, 0.90) + max(2.0 * median_particle_radius, 3.0 * 1.4826 * mad)
            outliers = [item for item in group if item[1] > upper_limit]
            outlier_particle_points.extend(tuple(map(float, points[item[3]])) for item in outliers)
            valid_group = [item for item in group if item[1] <= upper_limit]
            if not valid_group:
                valid_group = group
            selected = max(valid_group, key=lambda item: item[1])
            radial_center, surface_center_height, particle_radius, point_index = selected
            center_envelope.append((radial_center, surface_center_height, particle_radius))
            selected_particle_points.append(tuple(map(float, points[point_index])))

        if len(center_envelope) < min_points_per_ray:
            raise ValueError(f"insufficient radial support in direction {direction_index + 1}")
        center_slope = _linear_slope([(radius, height) for radius, height, _ in center_envelope])
        normal_length = math.hypot(center_slope, 1.0)
        surface_envelope = [
            (radius - center_slope * particle_radius / normal_length,
             height + particle_radius / normal_length)
            for radius, height, particle_radius in center_envelope
        ]

        # Reject isolated selected samples only when they depart from the robust
        # median pairwise trend by more than two median particle radii.
        pair_slopes = [
            (b[1] - a[1]) / (b[0] - a[0])
            for i, a in enumerate(surface_envelope)
            for b in surface_envelope[i + 1:]
            if b[0] != a[0]
        ]
        initial_slope = statistics.median(pair_slopes) if pair_slopes else center_slope
        initial_intercept = statistics.median(y - initial_slope * radius for radius, y in surface_envelope)
        residual_limit = 2.0 * median_particle_radius
        outlier_profile_indices = [
            i for i, (radius, height) in enumerate(surface_envelope)
            if abs(height - (initial_slope * radius + initial_intercept)) / math.hypot(initial_slope, 1.0) > residual_limit
        ]
        if len(surface_envelope) - len(outlier_profile_indices) >= min_points_per_ray:
            for i in outlier_profile_indices:
                outlier_particle_points.append(selected_particle_points[i])
            fit_envelope = [item for i, item in enumerate(surface_envelope) if i not in outlier_profile_indices]
            fit_center_envelope = [item for i, item in enumerate(center_envelope) if i not in outlier_profile_indices]
            fit_particle_points = [item for i, item in enumerate(selected_particle_points) if i not in outlier_profile_indices]
        else:
            fit_envelope = surface_envelope
            fit_center_envelope = center_envelope
            fit_particle_points = selected_particle_points
            outlier_profile_indices = []

        slope = _linear_slope(fit_envelope)
        angle = abs(math.degrees(math.atan(slope)))
        intercept = sum(y for _, y in fit_envelope) / len(fit_envelope) - slope * sum(r for r, _ in fit_envelope) / len(fit_envelope)
        direction_angles.append(angle)
        supports.append(len(fit_envelope))
        direction_profiles.append({
            "direction_index": direction_index,
            "angle_deg": angle,
            "envelope": fit_envelope,
            "center_envelope": [(radius, height) for radius, height, _ in fit_center_envelope],
            "selected_particle_points": fit_particle_points,
            "outlier_points": outlier_particle_points,
            "slope": slope,
            "intercept": intercept,
        })

    pair_profiles = []
    for axis_index in range(4):
        positive = direction_profiles[axis_index]
        negative = direction_profiles[axis_index + 4]
        pair_profiles.append({
            "axis_index": axis_index,
            "axis_angle_deg": axis_index * 45.0,
            "positive_angle_deg": positive["angle_deg"],
            "negative_angle_deg": negative["angle_deg"],
            "mean_angle_deg": (positive["angle_deg"] + negative["angle_deg"]) / 2.0,
            "positive_profile": positive,
            "negative_profile": negative,
            "positive_fit_points": positive["envelope"],
            "negative_fit_points": negative["envelope"],
            "positive_outlier_points": positive["outlier_points"],
            "negative_outlier_points": negative["outlier_points"],
        })
    mean = sum(direction_angles) / len(direction_angles)
    variance = sum((angle - mean) ** 2 for angle in direction_angles) / len(direction_angles)
    return {
        "mean_angle_deg": mean,
        "stddev_deg": math.sqrt(variance),
        "direction_angles_deg": direction_angles,
        "direction_support_bins": supports,
        "direction_profiles": direction_profiles,
        "pair_profiles": pair_profiles,
        "center_x": center_x,
        "center_z": center_z,
        "horizontal_reference_y": 0.0,
        "radial_bins": radial_bins,
        "median_particle_radius": median_particle_radius,
        "discarded_outliers": len(discarded_indices),
        "discarded_particle_points": [tuple(map(float, points[i])) for i in discarded_indices],
        "particle_cloud": [tuple(map(float, point)) for point in points],
    }


def measure_repose_eight_ray_least_squares(
    points,
    radial_bins=None,
    min_points_per_ray=8,
    particle_radii=None,
    height_band_factor=0.25,
    minimum_height_particle_diameters=2.0,
    outlier_tolerance_radii=1.75,
):
    """Measure eight repose faces from exact section surfaces, independently of discharge.

    The four vertical diametral planes pass through the fixed (x,z)=(0,0)
    axis at 0/45/90/135 degrees. A particle contributes to a plane only when
    its sphere intersects it; the in-plane radius is sqrt(r²-normal_offset²).
    Sample horizontal circle intersections on a regular physical-height grid.
    Sphere centers below the cutoff are ineligible. Fit a majority-supported
    free-intercept line; never force the fit through one selected particle.
    ``radial_bins`` is retained for API compatibility; it is not used here.
    """
    if len(points) < 2 * int(min_points_per_ray):
        raise UnmeasurableAngle("insufficient particles for four repose diameters")
    if particle_radii is None or len(particle_radii) != len(points):
        raise ValueError("particle_radii must contain one positive radius per repose particle")
    min_points_per_ray = int(min_points_per_ray)
    if min_points_per_ray < 5:
        raise ValueError("min_points_per_ray must be at least five base-to-top bands")
    height_band_factor = float(height_band_factor)
    if not math.isfinite(height_band_factor) or height_band_factor <= 0:
        raise ValueError("height_band_factor must be finite and positive")
    minimum_height_particle_diameters = float(minimum_height_particle_diameters)
    if not math.isfinite(minimum_height_particle_diameters) or minimum_height_particle_diameters < 0:
        raise ValueError("minimum_height_particle_diameters must be finite and nonnegative")
    outlier_tolerance_radii = float(outlier_tolerance_radii)
    if not math.isfinite(outlier_tolerance_radii) or outlier_tolerance_radii <= 0:
        raise ValueError("outlier_tolerance_radii must be finite and positive")
    cloud = [tuple(map(float, point)) for point in points]
    radii = [float(radius) for radius in particle_radii]
    if any(len(point) != 3 for point in cloud) or any(radius <= 0 or not math.isfinite(radius) for radius in radii):
        raise ValueError("repose points must be finite XYZ values and particle radii must be positive")
    if any(not math.isfinite(value) for point in cloud for value in point):
        raise ValueError("repose coordinates must be finite")
    radial_distances = [math.hypot(x, z) for x, _, z in cloud]
    sorted_radial_distances = sorted(radial_distances)
    radial_cap = 1.5 * sorted_radial_distances[round((len(sorted_radial_distances) - 1) * 0.75)]
    retained_indices = [index for index, radius in enumerate(radial_distances) if radius <= radial_cap]
    discarded_indices = [index for index, radius in enumerate(radial_distances) if radius > radial_cap]
    analysis_cloud = [cloud[index] for index in retained_indices]
    analysis_radii = [radii[index] for index in retained_indices]
    discarded_particle_points = [cloud[index] for index in discarded_indices]
    if len(analysis_cloud) < 2 * min_points_per_ray:
        raise UnmeasurableAngle("insufficient particles remain inside the robust pile domain")
    median_particle_radius = statistics.median(analysis_radii)
    minimum_slice_radius = 1e-12
    axis_angles = [0.0, 45.0, 90.0, 135.0]
    direction_profiles = []
    direction_angles = []
    direction_support = []
    pair_profiles = []
    slice_counts = []
    for axis_index, axis_angle in enumerate(axis_angles):
        theta = math.radians(axis_angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        transformed = []
        section_radii = []
        section_particle_indices = []
        for index, (x, y, z) in enumerate(analysis_cloud):
            horizontal = x * cos_t + z * sin_t
            normal = -x * sin_t + z * cos_t
            radius = analysis_radii[index]
            if abs(normal) >= radius:
                continue
            in_plane_radius = math.sqrt(max(radius * radius - normal * normal, 0.0))
            if in_plane_radius <= minimum_slice_radius:
                continue
            transformed.append((0.0, y, horizontal))
            section_radii.append(in_plane_radius)
            section_particle_indices.append(retained_indices[index])
        if len(transformed) < 2 * min_points_per_ray:
            raise UnmeasurableAngle(
                f"diametral section {axis_angle:.0f} deg has only {len(transformed)} particle-sphere intersections"
            )
        horizontal_extent = max(abs(point[2]) for point in transformed)
        if horizontal_extent <= 0:
            raise UnmeasurableAngle(f"diametral section {axis_angle:.0f} deg has no radial span")
        from optimizer.repose_profile import sample_repose_section
        try:
            measured = sample_repose_section(
                transformed, section_radii, section_particle_indices,
                median_particle_radius, height_band_factor,
                minimum_height_particle_diameters, min_points_per_ray)
        except ValueError as exc:
            raise UnmeasurableAngle(f"section {axis_angle:.0f}: {exc}") from exc
        slice_counts.append(len(transformed))

        def build_face(side, direction_index, orientation_deg):
            from optimizer.repose_profile import fit_repose_profile
            base_h, base_y = map(float, measured[f"base_{side}"])
            fit_points = [tuple(map(float, item)) for item in measured[f"{side}_fit_points"]]
            sample_ids = measured[f"{side}_sample_particle_indices"]
            fitted = fit_repose_profile(fit_points, median_particle_radius, sample_ids, outlier_tolerance_radii)
            valid = fitted["valid"] and len(fitted["inlier_points"]) >= min_points_per_ray
            reason = fitted["rejection_reason"] or ("insufficient inlier support" if not valid else "")
            if (side == "right" and fitted["horizontal_gradient"] >= 0) or (side == "left" and fitted["horizontal_gradient"] <= 0):
                valid, reason = False, "surface does not lean inward with height"
            angle = fitted["angle_deg"] if valid else None
            raw_slope = fitted["slope"]
            radial_slope = raw_slope if side == "right" else -raw_slope
            retained_points = fitted["inlier_points"]
            radial_envelope = [(abs(h), y) for h, y in retained_points]
            intercept = fitted["intercept"]
            start_y = max(float(measured[f"profile_start_height_{side}"]), min(y for _, y in retained_points))
            fit_start = ((start_y-intercept)/raw_slope, start_y)
            selected_centers = measured[f"{side}_selected_particle_centers"]
            projected_centers = [
                (float(h) * cos_t, float(y), float(h) * sin_t)
                for h, y in selected_centers
            ]
            downweighted_surface = fitted["downweighted_points"]
            projected_downweighted = [(h*cos_t, y, h*sin_t) for h,y in downweighted_surface]
            return {
                "direction_index": direction_index,
                "direction_angle_deg": orientation_deg % 360.0,
                "profile_selection": measured["profile_selection"],
                "angle_deg": angle,
                "valid": valid,
                "rejection_reason": reason,
                "slope": radial_slope,
                "raw_signed_slope": raw_slope,
                "intercept": intercept,
                "fit_start": fit_start,
                "base": (base_h, base_y),
                "base_axis_coordinate": float(measured[f"base_edge_axis_{side}"]),
                "profile_start_height": float(measured[f"profile_start_height_{side}"]),
                "fit_points": retained_points,
                "profile_points": fit_points,
                "all_samples_r2": fitted["all_samples_r2"],
                "envelope": radial_envelope,
                "center_envelope": [(abs(float(h)), float(y)) for h, y in selected_centers],
                "selected_particle_points": projected_centers,
                "selected_surface_points": [(h*cos_t, y, h*sin_t) for h,y in fit_points],
                "sample_particle_indices": measured[f"{side}_sample_particle_indices"],
                "sampling_evidence": measured[f"{side}_sampling_evidence"],
                "sample_heights": [y for _,y in fit_points],
                "axis_source": "outermost_section_sphere_surface_bound",
                "selected_particle_centers": selected_centers,
                "downweighted_points": downweighted_surface,
                "downweighted_particle_points": projected_downweighted,
                "outlier_points": [],
                "r2": float(fitted["r2"]),
                "r2_role": "diagnostic_y_on_horizontal_not_acceptance_gate",
                "orthogonal_rms_radii": fitted["orthogonal_rms_radii"],
                "orthogonal_quality": fitted["orthogonal_quality"],
                "fitted_horizontal_span_radii": fitted["fitted_horizontal_span_radii"],
                "vertical_span_radii": fitted["vertical_span_radii"],
                "departure_runs": fitted["departure_runs"],
                "fit_sample_weights": fitted["fit_sample_weights"],
                "excluded_sample_semantics": fitted["excluded_sample_semantics"],
                "validity_policy": fitted["validity_policy"],
                "fit_quality_warning": not fitted["valid"],
                "fit_support": len(retained_points),
                "candidate_count": int(measured[f"{side}_candidate_count"]),
                "downweighted_count": len(downweighted_surface),
                "rejected_count": len(downweighted_surface),
                "unique_particle_support": len({index for index,keep in zip(sample_ids,fitted["inlier_mask"]) if keep}),
                "median_section_radius": statistics.median(section_radii),
                "height_band_size": float(measured["height_bin_size"]),
            }

        positive = build_face("right", 2 * axis_index, axis_angle)
        negative = build_face("left", 2 * axis_index + 1, axis_angle + 180.0)
        direction_profiles.extend((positive, negative))
        direction_angles.extend((positive["angle_deg"], negative["angle_deg"]))
        direction_support.extend((positive["fit_support"], negative["fit_support"]))
        pair_profiles.append({
            "axis_index": axis_index,
            "axis_angle_deg": axis_angle,
            "positive_angle_deg": positive["angle_deg"],
            "negative_angle_deg": negative["angle_deg"],
            "mean_angle_deg": (positive["angle_deg"] + negative["angle_deg"]) / 2.0 if positive["valid"] and negative["valid"] else None,
            "positive_profile": positive,
            "negative_profile": negative,
            "positive_fit_points": positive["envelope"],
            "negative_fit_points": negative["envelope"],
            "positive_outlier_points": positive["downweighted_particle_points"],
            "negative_outlier_points": negative["downweighted_particle_points"],
            "section_particle_count": len(transformed),
            "section_projected_centers": [(h, y) for _, y, h in transformed],
            "section_particle_radii": section_radii,
            "section_particle_indices": section_particle_indices,
            "section_discarded_particle_points": [cloud[i] for i in discarded_indices if abs(-cloud[i][0]*sin_t+cloud[i][2]*cos_t) < radii[i]],
            "median_section_radius": statistics.median(section_radii),
        })
    all_valid = all(profile["valid"] for profile in direction_profiles)
    mean = sum(direction_angles) / len(direction_angles) if all_valid else None
    variance = sum((angle - mean) ** 2 for angle in direction_angles) / len(direction_angles) if all_valid else None
    result = {
        "profile_method": "repose_horizontal_surface_consensus",
        "outlier_tolerance_radii": outlier_tolerance_radii,
        "fit_weighting": "physical_height_trapezoidal_cells",
        "validity_policy": {**direction_profiles[0]["validity_policy"], "minimum_fit_samples": min_points_per_ray},
        "repose_axis_reference": "section_external_vertical",
        "height_sampling_method": "sphere_plane_horizontal_intersection",
        "height_band_factor": height_band_factor,
        "minimum_height_particle_diameters": minimum_height_particle_diameters,
        "slice_method": "particle_sphere_plane_intersection",
        "mean_angle_deg": mean,
        "stddev_deg": math.sqrt(variance) if variance is not None else None,
        "valid": all_valid,
        "direction_angles_deg": direction_angles,
        "direction_support_bins": direction_support,
        "direction_profiles": direction_profiles,
        "pair_profiles": pair_profiles,
        "axis_angles_deg": axis_angles,
        "slice_particle_counts": slice_counts,
        "center_x": 0.0,
        "center_z": 0.0,
        "horizontal_reference_y": 0.0,
        "median_particle_radius": median_particle_radius,
        "slice_half_width": None,
        "discarded_outliers": len(discarded_indices),
        "discarded_particle_points": discarded_particle_points,
        "particle_cloud": cloud,
    }
    if not all_valid:
        invalid = [f'{profile["direction_angle_deg"]:.0f}: {profile["rejection_reason"]}' for profile in direction_profiles if not profile["valid"]]
        exc = UnmeasurableAngle("; ".join(invalid))
        exc.diagnostic_result = result
        raise exc
    return result


def measure_v_sliding_angle_by_height(points, particle_radii, *, horizontal_axis, mesh_bounds=None, section_half_width=0.1, section_center=None, section_normal_bounds_sim=None, max_horizontal_jump_radii=5.0, min_fit_points=5, min_r2=0.90, profile_selection="nearest_centerline", height_band_factor=2.0, minimum_height_particle_diameters=0.0):
    """Trace two faces by diameter-height bands and fit all selected samples from base to top.

    ``nearest_centerline`` is the V-test selector; ``outer_envelope`` selects the exposed pile boundary for repose. Both share the same individual-radius surface correction, base anchor, Huber fit, and geometry checks.
    """
    if horizontal_axis not in {"x", "z"}:
        raise ValueError("horizontal_axis must be 'x' or 'z' because Y is vertical")
    if profile_selection not in {"nearest_centerline", "outer_envelope", "base_edge_nearest"}:
        raise ValueError("profile_selection must be 'nearest_centerline', 'outer_envelope', or 'base_edge_nearest'")
    if len(points) != len(particle_radii):
        raise ValueError("particle_radii must have one value per VTK particle")
    if section_half_width <= 0 or max_horizontal_jump_radii <= 0:
        raise ValueError("section_half_width and max_horizontal_jump_radii must be positive")
    if not math.isfinite(float(height_band_factor)) or float(height_band_factor) <= 0:
        raise ValueError("height_band_factor must be finite and positive")
    minimum_height_particle_diameters = float(minimum_height_particle_diameters)
    if not math.isfinite(minimum_height_particle_diameters) or minimum_height_particle_diameters < 0:
        raise ValueError("minimum_height_particle_diameters must be finite and nonnegative")
    if any(float(radius) <= 0 for radius in particle_radii):
        raise ValueError("particle radii must be positive")
    horizontal_index = 0 if horizontal_axis == "x" else 2
    normal_axis = "z" if horizontal_axis == "x" else "x"
    normal_index = 2 if horizontal_axis == "x" else 0
    mesh_bounds = mesh_bounds or {}
    for axis, bounds in mesh_bounds.items():
        if axis not in {"x", "z"} or len(bounds) != 2 or float(bounds[1]) <= float(bounds[0]):
            raise ValueError(f"invalid mesh bounds for axis {axis!r}")
    normal_bounds = mesh_bounds.get(normal_axis, (-section_half_width, section_half_width))
    if section_normal_bounds_sim is not None:
        section_normal_bounds_sim = tuple(map(float, section_normal_bounds_sim))
        if len(section_normal_bounds_sim) != 2 or not all(math.isfinite(value) for value in section_normal_bounds_sim) or section_normal_bounds_sim[1] <= section_normal_bounds_sim[0]:
            raise ValueError("section_normal_bounds_sim must contain two finite increasing bounds")
    horizontal_bounds = mesh_bounds.get(horizontal_axis, (0.0, 0.0))
    section_center = (
        (float(normal_bounds[0]) + float(normal_bounds[1])) / 2.0
        if section_center is None else float(section_center)
    )
    centerline = (float(horizontal_bounds[0]) + float(horizontal_bounds[1])) / 2.0
    section = [
        (float(point[horizontal_index]), float(point[1]), float(radius))
        for point, radius in zip(points, particle_radii)
        if (
            section_normal_bounds_sim[0] <= float(point[normal_index]) <= section_normal_bounds_sim[1]
            if section_normal_bounds_sim is not None
            else abs(float(point[normal_index]) - section_center) <= section_half_width
        )
    ]
    if len(section) < 2 * min_fit_points:
        raise ValueError("insufficient particles in the central vertical section")
    median_radius = statistics.median(radius for _, _, radius in section)
    band_height = float(height_band_factor) * median_radius
    if median_radius <= 0 or band_height <= 0:
        raise ValueError("invalid particle-diameter height band")

    def trace_side(side_name):
        side = [item for item in section if item[0] < centerline] if side_name == "left" else [item for item in section if item[0] > centerline]
        if not side:
            raise ValueError(f"no particles on {side_name} side of centerline")
        base_height = min(y for _, y, _ in side)
        top_height = max(y for _, y, _ in side)
        profile_start_height = base_height + minimum_height_particle_diameters * 2.0 * median_radius
        band_count = int(math.floor((top_height - base_height) / band_height)) + 1
        candidates = []
        candidate_groups = []
        for band_index in range(band_count):
            low = base_height + band_index * band_height
            high = low + band_height
            group = [item for item in side if low <= item[1] < high or (band_index == band_count - 1 and item[1] == high)]
            eligible_group = (
                [item for item in group if item[1] >= profile_start_height]
                if profile_selection == "nearest_centerline" else group
            )
            if eligible_group:
                # V-tests use the closest centerline point; repose uses the exposed outer face.
                if profile_selection == "nearest_centerline":
                    selected = min(eligible_group, key=lambda item: (abs(item[0] - centerline), abs(item[1] - (low + high) / 2)))
                elif side_name == "left":
                    selected = min(eligible_group, key=lambda item: (item[0], abs(item[1] - (low + high) / 2)))
                else:
                    selected = max(eligible_group, key=lambda item: (item[0], -abs(item[1] - (low + high) / 2)))
                candidates.append((band_index, selected))
                candidate_groups.append((band_index, eligible_group))
        if len(candidates) < min_fit_points:
            raise UnmeasurableAngle(f"insufficient height bands on {side_name} pile face")
        # Do not hard-reject a face for a large per-band horizontal change:
        # shallow but continuous faces can exceed this radius-scaled value.
        # Candidate outliers are screened by the robust residual and band-run test.
        kept = candidates
        surface_sign = -1.0 if side_name == "left" else 1.0
        # If the lowest band contains competing particles, use the robust trend
        # above it to avoid anchoring the face to a lone near-center particle.
        if len(candidates) >= min_fit_points + 1 and len(candidate_groups[0][1]) > 1:
            reference = [
                (z + surface_sign * radius, y)
                for _, (z, y, radius) in candidates[1:]
            ]
            pair_slopes = [
                (z2 - z1) / (y2 - y1)
                for index, (z1, y1) in enumerate(reference)
                for z2, y2 in reference[index + 1:]
                if y2 != y1
            ]
            if pair_slopes:
                outward_slope = statistics.median(pair_slopes)
                outward_intercept = statistics.median(
                    z - outward_slope * y for z, y in reference
                )
                base_band = candidate_groups[0][0]
                base_options = candidate_groups[0][1]
                ranked = []
                for z, y, radius in base_options:
                    surface_z = z + surface_sign * radius
                    orthogonal_error = abs(
                        surface_z - (outward_intercept + outward_slope * y)
                    ) / math.hypot(1.0, outward_slope)
                    ranked.append((orthogonal_error, y, abs(z - centerline), (z, y, radius)))
                best_error = min(item[0] for item in ranked)
                compatible = [
                    item for item in ranked
                    if item[0] <= best_error + 0.5 * median_radius
                ]
                error, _, _, replacement = min(
                    compatible,
                    key=lambda item: (item[1], item[2], item[3]),
                )
                if error <= 2.75 * median_radius:
                    candidates[0] = (base_band, replacement)
        surface_sign = -1.0 if side_name == "left" else 1.0
        if profile_selection == "base_edge_nearest":
            basal_reference_band, basal_reference = candidates[0]
            base_edge_axis = basal_reference[0]
            profile_start_height = base_height + minimum_height_particle_diameters * 2.0 * median_radius
            eligible_candidates = []
            for band, group in candidate_groups:
                eligible = [item for item in group if item[1] >= profile_start_height]
                if not eligible:
                    continue
                band_center = base_height + (band + 0.5) * band_height
                selected = min(
                    eligible,
                    key=lambda item: (abs(item[0] - base_edge_axis), abs(item[1] - band_center)),
                )
                eligible_candidates.append((band, selected))
            candidates = eligible_candidates
            if len(candidates) < min_fit_points:
                raise UnmeasurableAngle(
                    f"insufficient height bands above minimum repose height on {side_name} face"
                )
        else:
            base_edge_axis = candidates[0][1][0]
            if profile_selection != "nearest_centerline":
                profile_start_height = base_height
        kept = candidates
        surface_kept = [
            (band, z + surface_sign * radius, y, radius)
            for band, (z, y, radius) in candidates
        ]
        surface_discarded = []
        if profile_selection == "base_edge_nearest":
            base_band, (reference_z, reference_y, reference_radius) = basal_reference_band, basal_reference
            base_z = reference_z + surface_sign * reference_radius
            base_y = reference_y
        else:
            base_band, base_z, base_y, _ = surface_kept[0]
        residual_limit = 1.75 * median_radius

        def orthogonal_residual(sample, slope, intercept):
            _, z, y, _ = sample
            return abs(y - (intercept + slope * z)) / math.hypot(slope, 1.0)

        def residual(sample, slope):
            return orthogonal_residual(sample, slope, base_y - slope * base_z)

        pair_slopes = [
            (b[2] - a[2]) / (b[1] - a[1])
            for index, a in enumerate(surface_kept)
            for b in surface_kept[index + 1:]
            if b[1] != a[1]
        ]
        if not pair_slopes:
            raise UnmeasurableAngle(f"insufficient horizontal spread on {side_name} pile face")
        outward = [abs(sample[1] - centerline) for sample in surface_kept]
        smoothed = [
            statistics.median(outward[max(0, index - 1):min(len(outward), index + 2)])
            for index in range(len(outward))
        ]
        # Locate an actual extremum, not the last 2-of-3 sign-count window.
        # Require net directional support on both sides and prominence of one
        # particle diameter over the whole lobe. Local roughness is left to
        # Huber; a shallow but sustained return must not disappear merely
        # because its three-edge displacement is small. Keep raw samples so
        # median smoothing cannot erase short genuine returns.
        deltas = [b - a for a, b in zip(outward, outward[1:])]
        reversal_candidates = []
        for turn in range(2, len(outward) - 2):
            before = [value for value in deltas[:turn] if abs(value) > 1e-9 * median_radius][-3:]
            after = [value for value in deltas[turn:] if abs(value) > 1e-9 * median_radius][:3]
            net_before, net_after = sum(before), sum(after)
            peak_prominence = min(outward[turn] - min(outward[:turn]), outward[turn] - min(outward[turn + 1:]))
            trough_prominence = min(max(outward[:turn]) - outward[turn], max(outward[turn + 1:]) - outward[turn])
            if (before and before[-1] > 0 and deltas[turn] < -1e-9 * median_radius
                    and sum(value > 0 for value in before) >= 2
                    and sum(value < 0 for value in after) >= 2
                    and net_before > 0 > net_after
                    and peak_prominence >= 2.0 * median_radius):
                reversal_candidates.append((peak_prominence, turn + 1))
            if (before and before[-1] < 0 and deltas[turn] > 1e-9 * median_radius
                    and sum(value < 0 for value in before) >= 2
                    and sum(value > 0 for value in after) >= 2
                    and net_before < 0 < net_after
                    and trough_prominence >= 2.0 * median_radius):
                reversal_candidates.append((trough_prominence, turn + 1))
        # The SAME extremum may belong to either disjoint piece. Prefer the
        # established lower-piece ownership, but allow it to anchor the upper
        # piece when that alone restores minimum support (e.g. 20/4 -> 19/5).
        # This neither moves the physical turn nor duplicates a fit sample.
        reversal_candidates = [
            (strength, split - 1 if (
                min(split, len(surface_kept) - split) < min_fit_points
                and min(split - 1, len(surface_kept) - split + 1) >= min_fit_points
            ) else split)
            for strength, split in reversal_candidates
        ]
        # Prefer supported placement only for equally prominent extrema; never
        # hide a stronger true unsupported reversal behind a weaker supported
        # one or silently replace it by one line.
        strongest_turn = max(
            reversal_candidates,
            key=lambda item: (item[0], min(item[1], len(surface_kept) - item[1]) >= min_fit_points, -item[1]),
            default=(0.0, None),
        )
        split_at = strongest_turn[1] if profile_selection == "nearest_centerline" else None
        piece_samples = [surface_kept] if split_at is None else [surface_kept[:split_at], surface_kept[split_at:]]
        if split_at is not None and any(len(piece) < min_fit_points for piece in piece_samples):
            raise UnmeasurableAngle(f"insufficient segment support on {side_name} pile face for two-piece fit")
        if split_at is None and (len(surface_kept) < min_fit_points or surface_kept[-1][0] - base_band + 1 < min_fit_points):
            raise UnmeasurableAngle(f"insufficient base-to-top {side_name} face support")

        def robust_fit(fit_samples):
            fit_base_z, fit_base_y = fit_samples[0][1], fit_samples[0][2]
            weights = [1.0] * len(fit_samples)
            slopes = [(b[2] - a[2]) / (b[1] - a[1]) for i, a in enumerate(fit_samples) for b in fit_samples[i + 1:] if b[1] != a[1]]
            if not slopes:
                raise UnmeasurableAngle(f"insufficient horizontal spread on {side_name} profile segment")
            slope = statistics.median(slopes)
            def orthogonal_error(sample, candidate_slope):
                return abs(sample[2] - (fit_base_y + candidate_slope * (sample[1] - fit_base_z))) / math.hypot(candidate_slope, 1.0)
            for _ in range(12):
                denominator = sum(weight * (sample[1] - fit_base_z) ** 2 for sample, weight in zip(fit_samples, weights))
                if denominator <= 0:
                    raise UnmeasurableAngle(f"no horizontal spread on {side_name} profile segment")
                slope = sum(weight * (sample[1] - fit_base_z) * (sample[2] - fit_base_y) for sample, weight in zip(fit_samples, weights)) / denominator
                new_weights = [1.0 if orthogonal_error(sample, slope) <= residual_limit else residual_limit / orthogonal_error(sample, slope) for sample in fit_samples]
                if max(abs(a - b) for a, b in zip(weights, new_weights)) < 1e-6:
                    weights = new_weights
                    break
                weights = new_weights
            fitted_y = [fit_base_y + slope * (sample[1] - fit_base_z) for sample in fit_samples]
            mean_y = sum(sample[2] for sample in fit_samples) / len(fit_samples)
            ss_tot = sum((sample[2] - mean_y) ** 2 for sample in fit_samples)
            ss_res = sum((sample[2] - fit_y) ** 2 for sample, fit_y in zip(fit_samples, fitted_y))
            r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
            return {"angle_deg": abs(math.degrees(math.atan(slope))), "slope": slope, "r2": r2,
                    "base": (fit_base_z, fit_base_y), "fit_points": [(sample[1], sample[2]) for sample in fit_samples],
                    "fit_support": len(fit_samples), "fit_band_span": fit_samples[-1][0] - fit_samples[0][0] + 1,
                    "downweighted_points": [(sample[1], sample[2]) for sample, weight in zip(fit_samples, weights) if weight < 0.999],
                    "max_orthogonal_residual": max(orthogonal_error(sample, slope) for sample in fit_samples),
                    "fit_quality_warning": r2 < min_r2}
        fitted_segments = [robust_fit(piece) for piece in piece_samples]
        fit_samples = surface_kept
        slope = fitted_segments[0]["slope"] if len(fitted_segments) == 1 else sum(item["slope"] for item in fitted_segments) / len(fitted_segments)
        r2 = sum(item["r2"] for item in fitted_segments) / len(fitted_segments)
        robust_downweighted = [point for item in fitted_segments for point in item["downweighted_points"]]
        return {
            "angle_deg": sum(item["angle_deg"] for item in fitted_segments) / len(fitted_segments),
            "slope": slope, "r2": r2, "base": (base_z, base_y),
            "base_edge_axis": base_edge_axis, "profile_start_height": profile_start_height,
            "fit_points": [(z, y) for _, z, y, _ in fit_samples],
            "center_fit_points": [(z + surface_sign * radius, y) for _, (z, y, radius) in kept],
            "selected_particle_centers": [(z, y) for _, (z, y, _) in kept],
            "discarded_points": surface_discarded, "downweighted_points": robust_downweighted,
            "candidate_count": len(candidates), "fit_support": len(fit_samples),
            "fit_band_span": fit_samples[-1][0] - base_band + 1,
            "max_orthogonal_residual": max(item["max_orthogonal_residual"] for item in fitted_segments),
            "fit_quality_warning": any(item["fit_quality_warning"] for item in fitted_segments),
            "segments": fitted_segments, "segment_count": len(fitted_segments),
        }

    left = trace_side("left")
    right = trace_side("right")
    return {
        "horizontal_axis": horizontal_axis,
        "profile_selection": profile_selection,
        "centerline": centerline,
        "section_center": section_center,
        "section_half_width": section_half_width,
        "section_normal_bounds_sim": section_normal_bounds_sim,
        "section_particle_count": len(section),
        "height_bin_size": band_height,
        "median_radius": median_radius,
        "jump_limit": max_horizontal_jump_radii * median_radius,
        "left_angle_deg": left["angle_deg"],
        "right_angle_deg": right["angle_deg"],
        "left_slope": left["slope"],
        "right_slope": right["slope"],
        "mean_angle_deg": (left["angle_deg"] + right["angle_deg"]) / 2.0,
        "asymmetry_deg": abs(left["angle_deg"] - right["angle_deg"]),
        "left_r2": left["r2"],
        "right_r2": right["r2"],
        "left_fit_quality_warning": left["fit_quality_warning"],
        "right_fit_quality_warning": right["fit_quality_warning"],
        "base_left": left["base"],
        "base_right": right["base"],
        "base_edge_axis_left": left["base_edge_axis"],
        "base_edge_axis_right": right["base_edge_axis"],
        "profile_start_height_left": left["profile_start_height"],
        "profile_start_height_right": right["profile_start_height"],
        "minimum_height_particle_diameters": minimum_height_particle_diameters,
        "left_fit_points": left["fit_points"],
        "right_fit_points": right["fit_points"],
        "left_selected_particle_centers": left["selected_particle_centers"],
        "right_selected_particle_centers": right["selected_particle_centers"],
        "left_discarded_points": left["discarded_points"],
        "right_discarded_points": right["discarded_points"],
        "left_downweighted_points": left["downweighted_points"],
        "right_downweighted_points": right["downweighted_points"],
        "left_candidate_count": left["candidate_count"],
        "right_candidate_count": right["candidate_count"],
        "left_segments": left["segments"],
        "right_segments": right["segments"],
        "left_segment_count": left["segment_count"],
        "right_segment_count": right["segment_count"],
    }


def measure_v_sliding_angle_three_cuts(
    points,
    particle_radii,
    *,
    horizontal_axis,
    mesh_bounds=None,
    nbrratio,
    physical_centers=(-0.1, 0.0, 0.1),
    physical_half_width=0.075,
    minimum_height_particle_diameters=2.0,
):
    """Measure the three requested physical cut centers in NBR-scaled DEM units."""
    nbrratio = float(nbrratio)
    physical_centers = tuple(float(center) for center in physical_centers)
    physical_half_width = float(physical_half_width)
    if not math.isfinite(nbrratio) or nbrratio <= 0:
        raise ValueError("nbrratio must be a finite positive value")
    minimum_height_particle_diameters = float(minimum_height_particle_diameters)
    if not math.isfinite(minimum_height_particle_diameters) or minimum_height_particle_diameters < 0:
        raise ValueError("minimum_height_particle_diameters must be finite and nonnegative")
    if len(physical_centers) != 3 or len(set(physical_centers)) != 3:
        raise ValueError("three distinct physical cut centers are required")
    if any(not math.isfinite(center) for center in physical_centers):
        raise ValueError("physical cut centers must be finite")
    if not math.isfinite(physical_half_width) or physical_half_width <= 0:
        raise ValueError("physical_half_width must be a finite positive value")
    section_half_width = physical_half_width * nbrratio
    normal_axis = "x" if horizontal_axis == "z" else "z"
    normal_bounds = (mesh_bounds or {}).get(normal_axis)
    sections = []
    for physical_center in physical_centers:
        section_center = physical_center * nbrratio
        requested_band = (section_center - section_half_width, section_center + section_half_width)
        if normal_bounds is None:
            effective_band = requested_band
            band_clipped = False
        else:
            mesh_low, mesh_high = map(float, normal_bounds)
            effective_band = (max(requested_band[0], mesh_low), min(requested_band[1], mesh_high))
            if effective_band[0] >= effective_band[1]:
                raise ValueError(f"discharge cut x={physical_center:+.3f} m lies outside the mesh")
            band_clipped = not (
                math.isclose(effective_band[0], requested_band[0], rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(effective_band[1], requested_band[1], rel_tol=1e-9, abs_tol=1e-9)
            )
        measurement = measure_v_sliding_angle_by_height(
            points,
            particle_radii,
            horizontal_axis=horizontal_axis,
            mesh_bounds=mesh_bounds,
            section_half_width=section_half_width,
            section_center=section_center,
            section_normal_bounds_sim=effective_band,
            minimum_height_particle_diameters=minimum_height_particle_diameters,
        )
        measurement["section_center_physical"] = physical_center
        measurement["section_half_width_physical"] = physical_half_width
        measurement["requested_band_sim"] = requested_band
        measurement["effective_band_sim"] = effective_band
        measurement["effective_band_physical"] = (effective_band[0] / nbrratio, effective_band[1] / nbrratio)
        measurement["mesh_normal_bounds"] = tuple(map(float, normal_bounds)) if normal_bounds is not None else None
        measurement["band_clipped"] = band_clipped
        measurement["nbrratio"] = nbrratio
        sections.append(measurement)
    left_angle = sum(item["left_angle_deg"] for item in sections) / len(sections)
    right_angle = sum(item["right_angle_deg"] for item in sections) / len(sections)
    return {
        "horizontal_axis": horizontal_axis,
        "centerline": sections[0]["centerline"],
        "section_half_width": section_half_width,
        "section_half_width_physical": physical_half_width,
        "nbrratio": nbrratio,
        "section_centers_physical": list(physical_centers),
        "minimum_height_particle_diameters": minimum_height_particle_diameters,
        "left_angle_deg": left_angle,
        "right_angle_deg": right_angle,
        "mean_angle_deg": sum(item["mean_angle_deg"] for item in sections) / len(sections),
        "asymmetry_deg": abs(left_angle - right_angle),
        "sections": sections,
    }


def measure_v_sliding_angle(points, *, horizontal_axis, bins=72, section_half_width=0.1, surface_quantile=0.75, min_points_per_bin=2, min_r2=0.90, mesh_bounds=None):
    """Mede as duas faces no plano horizontal escolhido (X ou Z) versus Y vertical."""
    if horizontal_axis not in {"x", "z"}:
        raise ValueError("horizontal_axis must be 'x' or 'z' because Y is vertical")
    if section_half_width <= 0:
        raise ValueError("section_half_width must be positive")
    horizontal_index = 0 if horizontal_axis == "x" else 2
    normal_axis = "z" if horizontal_axis == "x" else "x"
    normal_index = 2 if horizontal_axis == "x" else 0
    mesh_bounds = mesh_bounds or {}
    for axis, bounds in mesh_bounds.items():
        if axis not in {"x", "z"} or len(bounds) != 2 or float(bounds[1]) <= float(bounds[0]):
            raise ValueError(f"invalid mesh bounds for axis {axis!r}")
    normal_bounds = mesh_bounds.get(normal_axis, (-section_half_width, section_half_width))
    horizontal_bounds = mesh_bounds.get(horizontal_axis, (0.0, 0.0))
    section_center = (float(normal_bounds[0]) + float(normal_bounds[1])) / 2.0
    horizontal_center = (float(horizontal_bounds[0]) + float(horizontal_bounds[1])) / 2.0
    section = [(point[horizontal_index], point[1]) for point in points if abs(point[normal_index] - section_center) <= section_half_width]
    if len(section) < 2 * min_points_per_bin * 4:
        raise ValueError(f"insufficient particles in the central {horizontal_axis.upper()}-Y section")
    x_min, x_max = min(x for x, _ in section), max(x for x, _ in section)
    if x_max <= x_min:
        raise ValueError(f"zero {horizontal_axis.upper()} span in sliding section")
    width = (x_max - x_min) / bins
    envelope = []
    for index in range(bins):
        low, high = x_min + index * width, x_min + (index + 1) * width
        group = [(x, y) for x, y in section if low <= x < high or (index == bins - 1 and x == high)]
        if len(group) >= min_points_per_bin:
            surface_x, surface_y = sorted(group, key=lambda point: point[1])[round((len(group) - 1) * surface_quantile)]
            envelope.append((surface_x, surface_y))
    left = [(x, y) for x, y in envelope if x < horizontal_center]
    right = [(x, y) for x, y in envelope if x > horizontal_center]
    if len(left) < 3 or len(right) < 3:
        raise ValueError("insufficient populated bins on both sliding pile faces")

    def regression(points):
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator <= 0:
            raise ValueError("zero horizontal spread in sliding face profile")
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
        intercept = mean_y - slope * mean_x
        residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
        return slope, intercept, residual

    def fit(side):
        # Anchor each face at its lowest free-surface envelope point (the toe).
        # Ties use the point farthest from the center, i.e. the discharge-side end.
        toe = min(side, key=lambda point: (point[1], -abs(point[0] - horizontal_center)))
        ordered = sorted(side, key=lambda point: abs(point[0] - toe[0]))

        def anchored_slope(points):
            denominator = sum((x - toe[0]) ** 2 for x, _ in points)
            if denominator <= 0:
                raise ValueError("insufficient spread from sliding-face toe")
            return sum((x - toe[0]) * (y - toe[1]) for x, y in points) / denominator

        def anchored_residual(points, slope):
            return sum((y - (toe[1] + slope * (x - toe[0]))) ** 2 for x, y in points)

        whole_slope = anchored_slope(ordered)
        whole_residual = anchored_residual(ordered, whole_slope)
        fit_points = ordered
        fit_end_distance = abs(ordered[-1][0] - toe[0])
        best_bic = float("inf")
        n = len(ordered)
        if n >= 7:
            # Compare a single toe-anchored face with a two-segment profile:
            # sloped face from the toe, then a separate upper segment. BIC
            # penalizes the extra segment/change point, avoiding an arbitrary
            # fixed angle threshold for deciding what counts as a flat top.
            single_bic = n * math.log(max(whole_residual / n, 1e-24)) + math.log(n)
            for split in range(3, n - 2):
                ramp = ordered[:split]
                upper = ordered[split:]
                ramp_slope = anchored_slope(ramp)
                upper_slope, upper_intercept, upper_residual = regression(upper)
                if abs(upper_slope) >= abs(ramp_slope):
                    continue
                residual = anchored_residual(ramp, ramp_slope) + upper_residual
                # 4 fitted quantities: ramp slope, upper slope/intercept,
                # and the discrete change-point location.
                bic = n * math.log(max(residual / n, 1e-24)) + 4 * math.log(n)
                if bic < best_bic:
                    best_bic = bic
                    candidate_end = ramp[-1][0]
                    candidate_points = ramp
                    candidate_slope = ramp_slope
            if best_bic < single_bic:
                fit_points = candidate_points
                fit_end_distance = abs(candidate_end - toe[0])
                slope = candidate_slope
            else:
                slope = whole_slope
        else:
            slope = whole_slope

        fitted = [(x, toe[1] + slope * (x - toe[0])) for x, _ in fit_points]
        mean_y = sum(y for _, y in fit_points) / len(fit_points)
        ss_tot = sum((y - mean_y) ** 2 for _, y in fit_points)
        ss_res = sum((y - estimate) ** 2 for (_, y), (_, estimate) in zip(fit_points, fitted))
        r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        if r2 < min_r2:
            raise UnmeasurableAngle(f"poor linear fit on {horizontal_axis.upper()}-axis sliding pile face: R2={r2:.3f}")
        return abs(math.degrees(math.atan(slope))), r2, toe, fit_end_distance, fit_points, slope

    left_angle, left_r2, left_toe, left_end, left_fit_points, left_slope = fit(left)
    right_angle, right_r2, right_toe, right_end, right_fit_points, right_slope = fit(right)
    left_excluded_points = [point for point in left if point not in left_fit_points]
    right_excluded_points = [point for point in right if point not in right_fit_points]
    return {
        "horizontal_axis": horizontal_axis,
        "left_angle_deg": left_angle,
        "right_angle_deg": right_angle,
        "mean_angle_deg": (left_angle + right_angle) / 2,
        "asymmetry_deg": abs(left_angle - right_angle),
        "envelope_bins": len(envelope),
        "left_r2": left_r2,
        "right_r2": right_r2,
        "section_center": section_center,
        "horizontal_center": horizontal_center,
        "toe_left": left_toe,
        "toe_right": right_toe,
        "left_fit_end_distance": left_end,
        "right_fit_end_distance": right_end,
        "left_fit_points": left_fit_points,
        "right_fit_points": right_fit_points,
        "left_excluded_points": left_excluded_points,
        "right_excluded_points": right_excluded_points,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "bins": bins,
        "surface_quantile": surface_quantile,
    }


def measure_drawdown_inner_faces(points, bins=None, surface_quantile=1.0):
    bins = bins or max(8, min(12, len(points) // 30))
    z_min, z_max = min(p[2] for p in points), max(p[2] for p in points); width = (z_max-z_min)/bins; profile=[]
    for i in range(bins):
        lo, hi = z_min+i*width, z_min+(i+1)*width
        group = [p for p in points if lo <= p[2] < hi or (i == bins-1 and p[2] == hi)]
        ys = sorted(p[1] for p in group)
        if ys: profile.append((sum(p[2] for p in group)/len(group),ys[round((len(ys)-1)*surface_quantile)]))
    left=[p for p in profile if p[0]<0]; right=[p for p in profile if p[0]>0]
    if len(left)<2 or len(right)<2: raise ValueError("insufficient drawdown profile bins")
    lp=max(left,key=lambda p:p[1]); rp=max(right,key=lambda p:p[1])
    li=[p for p in left if p[0]>=lp[0]]; ri=[p for p in right if p[0]<=rp[0]]
    if len(li) < 3 or len(ri) < 3 or max(z for z, _ in li) - min(z for z, _ in li) < 0.008 or max(z for z, _ in ri) - min(z for z, _ in ri) < 0.008:
        raise ValueError("insufficient resolution on drawdown inner faces")
    la,ra=abs(math.degrees(math.atan(_linear_slope(li)))),abs(math.degrees(math.atan(_linear_slope(ri))))
    return {"left_angle_deg":la,"right_angle_deg":ra,"mean_angle_deg":(la+ra)/2,"asymmetry_deg":abs(la-ra),"envelope_bins":len(profile)}
