"""Local reimplementation from Rocky 2026 R1 manual, NOT the ANSYS scripts.

Missing cells are NaN (never zero or interpolated). Coordinates are particle
centres; radii set resolution only, not a radius-corrected free-surface contour.
See METODOLOGIA.md for documented steps versus explicit local decisions.
"""
from __future__ import annotations
import math
from numbers import Real
import numpy as np

METHOD_VERSION = 'rocky-manual-2026r1-local-v1'
# Provisional LOCAL v2 straight-face quality threshold, not a vendor value.
V2_MIN_R_SQUARED = .95


class UnmeasurableProfile(ValueError):
    """Insufficient observable geometry; no surrogate angle is returned."""


def _validate_minimum_fit_r_squared(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0. <= value <= 1.):
        raise ValueError('minimum_fit_r_squared must be a finite number in [0, 1]')
    return float(value)


def eulerian_maxima(points, *, horizontal_axis, depth_axis, horizontal_edges,
                     depth_edges, vertical_bounds):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('finite XYZ particle centres required')
    h, d = np.asarray(horizontal_edges, float), np.asarray(depth_edges, float)
    if len(h) < 2 or len(d) < 2 or not (np.diff(h) > 0).all() or not (np.diff(d) > 0).all():
        raise ValueError('strictly increasing cell edges required')
    if {horizontal_axis, depth_axis} != {0, 2}:
        raise ValueError('Y is vertical; horizontal/depth axes must be X/Z')
    lo, hi = vertical_bounds
    mask = ((p[:, 1] >= lo) & (p[:, 1] <= hi) &
            (p[:, horizontal_axis] >= h[0]) & (p[:, horizontal_axis] <= h[-1]) &
            (p[:, depth_axis] >= d[0]) & (p[:, depth_axis] <= d[-1]))
    q = p[mask]
    i = np.minimum(np.searchsorted(h, q[:, horizontal_axis], side='right') - 1, len(h)-2)
    j = np.minimum(np.searchsorted(d, q[:, depth_axis], side='right') - 1, len(d)-2)
    maxima = np.full((len(h)-1, len(d)-1), -np.inf)
    counts = np.zeros(maxima.shape, dtype=int)
    np.maximum.at(maxima, (i, j), q[:, 1])
    np.add.at(counts, (i, j), 1)
    maxima[counts == 0] = np.nan
    return {'maxima': maxima, 'counts': counts, 'particles_in_roi': int(mask.sum()),
            'particles_outside_roi': int((~mask).sum())}


def profile_statistics(maxima):
    a = np.asarray(maxima, float)
    support = np.isfinite(a).sum(axis=1)
    average = np.full(a.shape[0], np.nan)
    minimum, maximum = average.copy(), average.copy()
    valid = support > 0
    average[valid] = np.nansum(a[valid], axis=1) / support[valid]
    minimum[valid] = np.min(np.where(np.isfinite(a[valid]), a[valid], np.inf), axis=1)
    maximum[valid] = np.max(np.where(np.isfinite(a[valid]), a[valid], -np.inf), axis=1)
    return {'average': average, 'minimum': minimum, 'maximum': maximum, 'support': support}


def treat_saor(x, stats, *, diameter, floor, mode='strict_v1'):
    """Select observed mean-surface cells; strict_v1 remains reproducible.

    mean_profile_v2 is an explicit LOCAL method revision, not an ANSYS claim.
    It keeps finite observed means above the same one-diameter floor, without
    pruning them by extrema of different orientations/depth cells. In
    particular, small reversals of the mean are not silently cherry-picked
    away. Means are neither interpolated nor replaced with fitted values.
    """
    if mode not in {'strict_v1', 'mean_profile_v2'}:
        raise ValueError('unknown profile treatment mode')
    means, lows = stats['average'], stats['minimum']
    if mode == 'mean_profile_v2':
        reasons = ['empty_bin' if not np.isfinite(y) else
                   'mean_not_above_one_diameter' if y-floor <= diameter else
                   'retained' for y in means]
        return {'indices': np.flatnonzero(np.asarray(reasons) == 'retained'),
                'reasons': reasons}
    indices, reasons = [], []
    previous_mean = previous_low = -np.inf
    for i in range(len(x)):
        if not np.isfinite(means[i]) or not np.isfinite(lows[i]):
            reason = 'empty_bin'
        elif means[i] - floor <= diameter:
            reason = 'mean_not_above_one_diameter'
        elif lows[i] < previous_low:
            reason = 'minimum_decreases_toward_centre'
        elif lows[i] <= previous_mean:
            reason = 'minimum_not_above_previous_retained_means'
        else:
            reason = 'retained'
            indices.append(i)
            previous_mean = max(previous_mean, means[i])
            previous_low = lows[i]
        reasons.append(reason)
    return {'indices': np.asarray(indices, int), 'reasons': reasons}


def progressive_fit(x, y, *, min_points=5):
    """Unanchored OLS y=m*x+b on prefixes 5..N; maximum R² wins.

    Equal scores keep the first prefix (explicit deterministic local tie rule).
    A flat profile has undefined correlation, not a perfect angle observation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < min_points or min_points < 5:
        raise UnmeasurableProfile('at least five profile points required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise UnmeasurableProfile('non-finite treated profile')
    candidates = []
    for n in range(min_points, len(x)+1):
        xx, yy = x[:n], y[:n]
        dx, dy = xx-xx.mean(), yy-yy.mean()
        denominator = float(dx @ dx)
        total = float(dy @ dy)
        if denominator <= 0 or total <= np.finfo(float).eps * max(float(yy @ yy), 1e-300):
            continue
        slope = float(dx @ dy) / denominator
        intercept = float(yy.mean() - slope * xx.mean())
        residual = float(np.sum((yy-(slope*xx+intercept))**2))
        r2 = 1. - residual / total
        candidates.append({'slope': slope, 'intercept': intercept, 'r_squared': r2,
                           'r_value': math.copysign(math.sqrt(max(0., r2)), slope),
                           'points_used': n, 'fit_angle_deg': math.degrees(math.atan(abs(slope)))})
    if not candidates:
        raise UnmeasurableProfile('constant or degenerate profile; no valid regression')
    best = max(candidates, key=lambda a: a['r_squared'])
    return {**best, 'candidates': candidates}


def _particles(points, radii):
    p, r = np.asarray(points, float), np.asarray(radii, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError('nonempty Nx3 particle array required')
    if r.shape != (len(p),) or not np.isfinite(p).all() or not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError('finite centres and one positive radius per particle required')
    return p, r, float(2.*r.mean())


def measure_saor(points, radii, *, center_xz, width, vertical_bounds,
                 tray_half_width, divisions_factor=1.5, number_of_samples=36,
                 treatment_mode='strict_v1', minimum_fit_r_squared=V2_MIN_R_SQUARED):
    minimum_fit_r_squared = _validate_minimum_fit_r_squared(minimum_fit_r_squared)
    p, r, diameter = _particles(points, radii)
    if width <= 0 or tray_half_width <= 0 or divisions_factor <= 0 or number_of_samples != 36:
        raise ValueError('positive ROI, factor and exactly 36 samples required')
    if vertical_bounds[1] <= vertical_bounds[0]:
        raise ValueError('positive vertical extent required')
    # Even bin count prevents a cell from crossing the symmetry centre.
    # Keep the sampling slab and floor treatment unchanged; refine grid only.
    resolution_diameter = float(2. * r.min())
    n = max(2, 2*math.ceil(width/(resolution_diameter*divisions_factor)/2))
    edges = np.linspace(-width/2, width/2, n+1)
    coordinates = (edges[:-1]+edges[1:])/2
    maxima, counts, roi_counts = [], [], []
    for degrees in np.arange(36)*10.:
        t = math.radians(degrees)
        q = p.copy()
        dx, dz = p[:, 0]-center_xz[0], p[:, 2]-center_xz[1]
        q[:, 0], q[:, 2] = dx*math.cos(t)+dz*math.sin(t), -dx*math.sin(t)+dz*math.cos(t)
        grid = eulerian_maxima(q, horizontal_axis=0, depth_axis=2,
                              horizontal_edges=edges, depth_edges=[-diameter/2, diameter/2],
                              vertical_bounds=vertical_bounds)
        maxima.append(grid['maxima'][:, 0]); counts.append(grid['counts'][:, 0])
        roi_counts.append(grid['particles_in_roi'])
    maxima, counts = np.asarray(maxima).T, np.asarray(counts).T
    stats = profile_statistics(maxima)
    # Full-circle sweep is symmetric in the ideal case. As in the SAOR
    # example, fit one outer-to-centre half of the averaged pile.
    half = coordinates < 0
    half_stats = {k: v[half] for k, v in stats.items()}
    treatment = treat_saor(coordinates[half], half_stats, diameter=diameter,
                           floor=vertical_bounds[0], mode=treatment_mode)
    selected = np.flatnonzero(half)[treatment['indices']]
    base = {'method': METHOD_VERSION if treatment_mode=='strict_v1' else 'mean-profile-local-v2',
            'treatment_mode':treatment_mode, 'kind': 'SAOR', 'samples': 36,
            'mean_diameter_m': diameter, 'coordinates': coordinates, 'statistics': stats,
            'grid_maxima': maxima, 'grid_counts': counts, 'horizontal_edges': edges,
            'sample_degrees': np.arange(36)*10., 'roi_particle_counts': roi_counts,
            'treated_indices': selected, 'treatment_reasons': treatment['reasons'],
            'treatment_scope': 'negative_half_outer_to_centre', 'cell_width_m': width/n,
            'resolution_diameter_m': resolution_diameter,
            'resolution_diameter_basis': 'minimum_particle_diameter',
            'target_cell_size_m': resolution_diameter*divisions_factor,
            'slab_width_m': diameter, 'center_xz': center_xz, 'floor_m': vertical_bounds[0],
            'tray_half_width_m': tray_half_width, 'equivalence_to_ansys_verified': False,
            'minimum_fit_r_squared':minimum_fit_r_squared if treatment_mode=='mean_profile_v2' else None}
    fit = None
    try:
        fit = progressive_fit(coordinates[selected], stats['average'][selected])
        if fit['slope'] <= 0:
            raise UnmeasurableProfile('SAOR must rise towards centre')
        if treatment_mode=='mean_profile_v2' and fit['r_squared'] < minimum_fit_r_squared:
            raise UnmeasurableProfile(f'mean profile does not support a straight face: R2 below {minimum_fit_r_squared}')
    except UnmeasurableProfile as exc:
        return {**base, 'status': 'unmeasurable', 'reason': str(exc),
                'candidate_fit':fit, 'mean_angle_deg': None}
    toe = (vertical_bounds[0]-fit['intercept'])/fit['slope']
    outside = abs(toe) > tray_half_width + 1e-10*max(width, 1e-9)
    return {**base, 'status': 'measured', 'fit': fit, 'fit_indices': selected[:fit['points_used']],
            'mean_angle_deg': 90. if outside else fit['fit_angle_deg'],
            'tray_intersection_m': toe, 'tray_intersection_outside': outside,
            'ninety_degree_rule_applied': outside}


def measure_daor(points, radii, *, bounds, divisions_factor=1.5, response_convention='left_face',
                 treatment_mode='strict_v1', minimum_fit_r_squared=V2_MIN_R_SQUARED):
    """Width/depth Eulerian maxima, averaged over DEPTH, at one snapshot.

    DAOR treatment is a declared local convention: each half is scanned from
    its lowest mean cell towards the outer wall, then rising-profile treatment
    and progressive OLS are applied. The manual's broken cross-reference does
    NOT establish that this is the original DAOR selection code.
    """
    minimum_fit_r_squared = _validate_minimum_fit_r_squared(minimum_fit_r_squared)
    p, r, diameter = _particles(points, radii)
    if treatment_mode not in {'strict_v1', 'mean_profile_v2'}:
        raise ValueError('unknown profile treatment mode')
    if divisions_factor <= 0 or response_convention not in {'left_face', 'right_face', 'paired_mean'}:
        raise ValueError('positive factor and supported response convention required')
    if any(not np.isfinite(bounds[a]).all() or bounds[a][1] <= bounds[a][0] for a in ['x','y','z']):
        raise ValueError('finite positive container bounds required')
    width, depth = bounds['z'][1]-bounds['z'][0], bounds['x'][1]-bounds['x'][0]
    # DAOR grid reference: smallest particle, not the mean diameter.
    # Preserve the mean diameter for the existing floor/profile treatment.
    resolution_diameter = float(2. * r.min())
    n = max(2, 2*math.ceil(width/(resolution_diameter*divisions_factor)/2))
    nd = max(1, math.ceil(depth/(resolution_diameter*divisions_factor) - 1e-12))
    he = np.linspace(*bounds['z'], n+1)
    de = np.linspace(*bounds['x'], nd+1)
    grid = eulerian_maxima(p, horizontal_axis=2, depth_axis=0, horizontal_edges=he,
                          depth_edges=de, vertical_bounds=bounds['y'])
    x = (he[:-1]+he[1:])/2
    stats = profile_statistics(grid['maxima'])
    centre = sum(bounds['z'])/2
    faces = {}
    for side in ['left', 'right']:
        indices = np.flatnonzero(x < centre if side == 'left' else x > centre)
        if side == 'left':
            indices = indices[::-1]  # inner valley -> outer wall
        finite = indices[np.isfinite(stats['average'][indices])]
        if not len(finite):
            faces[side] = {'status':'unmeasurable', 'reason':'no occupied cells in face', 'fit':None}
            continue
        # The lowest mean fixes the start; never fit across a central empty gap.
        start = int(np.argmin(stats['average'][finite]))
        origin = finite[start]
        ordered = indices[np.flatnonzero(indices == origin)[0]:]
        st = {k:v[ordered] for k,v in stats.items()}
        treatment = treat_saor(x[ordered], st, diameter=diameter, floor=bounds['y'][0], mode=treatment_mode)
        treated = ordered[treatment['indices']]
        face = {'ordered_indices':ordered, 'treated_indices':treated,
                'treatment_reasons':treatment['reasons'], 'fit':None,
                'treatment_direction':'inner_minimum_to_outer_wall'}
        fit = None
        try:
            fit = progressive_fit(x[treated], stats['average'][treated])
            if (side == 'left' and fit['slope'] >= 0) or (side == 'right' and fit['slope'] <= 0):
                raise UnmeasurableProfile('face slope does not face the central void')
            if treatment_mode=='mean_profile_v2' and fit['r_squared'] < minimum_fit_r_squared:
                raise UnmeasurableProfile(f'mean profile does not support a straight face: R2 below {minimum_fit_r_squared}')
            faces[side] = {**face, 'status':'measured', 'fit':fit,
                           'fit_indices':treated[:fit['points_used']]}
        except UnmeasurableProfile as exc:
            faces[side] = {**face, 'status':'unmeasurable', 'reason':str(exc), 'candidate_fit':fit}
    angles = {k:(v['fit']['fit_angle_deg'] if v['status']=='measured' else None) for k,v in faces.items()}
    mean = None if any(a is None for a in angles.values()) else sum(angles.values())/2
    response = {'left_face':angles['left'], 'right_face':angles['right'], 'paired_mean':mean}[response_convention]
    return {'method':METHOD_VERSION if treatment_mode=='strict_v1' else 'mean-profile-local-v2',
            'treatment_mode':treatment_mode, 'kind':'DAOR', 'status':'measured' if response is not None else 'unmeasurable',
            'reason':None if response is not None else 'selected response has no measurable face',
            'response_angle_deg':response, 'response_convention':response_convention,
            'left_angle_deg':angles['left'], 'right_angle_deg':angles['right'], 'paired_mean_angle_deg':mean,
            'asymmetry_deg':None if mean is None else abs(angles['left']-angles['right']),
            'faces':faces, 'coordinates':x, 'statistics':stats, 'grid_maxima':grid['maxima'],
            'grid_counts':grid['counts'], 'horizontal_edges':he, 'depth_edges':de, 'bounds':bounds,
            'particles_in_roi':grid['particles_in_roi'], 'particles_outside_roi':grid['particles_outside_roi'],
            'mean_diameter_m':diameter, 'cell_width_m':width/n, 'cell_depth_m':depth/nd,
            'resolution_diameter_m':resolution_diameter,
            'resolution_diameter_basis':'minimum_particle_diameter',
            'target_cell_size_m':resolution_diameter*divisions_factor,
            'equivalence_to_ansys_verified':False, 'daor_sanitization_is_local_convention':True,
            'minimum_fit_r_squared':minimum_fit_r_squared if treatment_mode=='mean_profile_v2' else None}
