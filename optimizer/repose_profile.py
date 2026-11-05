"""Repose-only robust fitting; discharge's anchored estimator is unchanged."""
import math
import statistics
from collections import Counter


def sample_repose_section(points, plane_radii, particle_indices, median_radius,
                          height_band_factor, minimum_height_particle_diameters,
                          min_fit_points):
    """Trace the exact exterior of the section-circle union at regular heights.

    Only sphere centers above the declared cutoff are eligible. At each height,
    rank the actual horizontal sphere/plane/height intersections by distance to
    a lateral axis outside the whole section, never by a center-only height bin.
    """
    section = [(index, h, y, radius) for index, (_, y, h), radius
               in zip(particle_indices, points, plane_radii)]
    spacing = height_band_factor * median_radius
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('invalid physical height spacing')
    result = {'median_radius': median_radius, 'height_bin_size': spacing,
              'profile_selection': 'base_edge_nearest'}
    for side, sign in (('left', -1), ('right', 1)):
        particles = [item for item in section if sign*(item[1]+sign*item[3]) > 0]
        if not particles:
            raise ValueError('no particles on '+side+' repose face')
        base_y = min(y for _,h,y,r in particles)
        cutoff = base_y + minimum_height_particle_diameters*2*median_radius
        eligible = [item for item in particles if item[2] >= cutoff]
        if not eligible:
            raise ValueError('no particles above repose cutoff on '+side)
        axis = max((h+sign*r for _,h,y,r in particles), key=lambda h: sign*h)
        top = max(y+r for _,h,y,r in eligible)
        samples, centers, ids, evidence = [], [], [], []
        level_count = (top-cutoff)/spacing
        if not math.isfinite(level_count) or level_count > 512:
            raise ValueError('height grid exceeds 512 levels; choose a physical sampling step')
        for band in range(int(math.floor(level_count))+1):
            height = cutoff+band*spacing
            candidates = []
            for index,h,y,radius in eligible:
                if abs(height-y) > radius:
                    continue
                surface = h+sign*math.sqrt(max(radius*radius-(height-y)**2,0.0))
                if sign*surface < 0:
                    continue
                candidates.append((abs(surface-axis),abs(y-height),index,surface,h,y,radius))
            if not candidates:
                continue
            distance,_,index,surface,h,y,radius = min(candidates)
            samples.append((surface,height))
            centers.append((h,y))
            ids.append(index)
            evidence.append({'height_index':band,'height':height,'particle_index':index,
                             'center':(h,y),'surface':(surface,height),
                             'section_radius':radius,'distance_to_axis':distance,
                             'candidate_count':len(candidates),
                             'eligible_candidates':[{'particle_index':c[2],
                              'surface_h':c[3],'distance_to_axis':c[0]} for c in candidates]})
        if len(samples) < 2:
            raise ValueError('insufficient sampled heights above cutoff on '+side)
        result.update({side+'_fit_points':samples, side+'_selected_particle_centers':centers,
                       side+'_sample_particle_indices':ids, side+'_sampling_evidence':evidence,
                       side+'_candidate_count':len(samples), 'base_'+side:(axis,base_y),
                       'base_edge_axis_'+side:axis,'profile_start_height_'+side:cutoff})
    return result


def fit_repose_profile(points, median_radius, particle_indices=None, residual_limit_factor=1.75):
    """Fit a free-intercept line using deterministic maximum consensus.

    Weights integrate physical height (trapezoidal cells), not particle votes.
    Repeated observations at an identical height share that cell's weight.
    Unique particle support is checked separately; repeated samples are not
    claimed as independent particles. Incompatible samples have zero fit weight.
    The radius-scaled tolerance is a declared project policy, not a standard.
    """
    samples = [(float(h), float(y)) for h, y in points]
    ids = list(range(len(samples))) if particle_indices is None else list(particle_indices)
    if len(ids) != len(samples):
        raise ValueError('one particle index is required per surface sample')
    if not math.isfinite(median_radius) or median_radius <= 0 or not math.isfinite(residual_limit_factor) or residual_limit_factor <= 0:
        raise ValueError('repose radius and residual tolerance must be finite and positive')
    if any(not math.isfinite(value) for point in samples for value in point):
        raise ValueError('repose samples must be finite')
    counts = Counter(y for h,y in samples)
    levels = sorted(counts)
    spans = {y:((levels[k+1] if k+1<len(levels) else y)-(levels[k-1] if k>0 else y))/2 for k,y in enumerate(levels)}
    weights = [spans[y]/counts[y] for h,y in samples]
    limit = residual_limit_factor * median_radius
    gradients = [(h2-h1)/(y2-y1) for i, (h1,y1) in enumerate(samples)
                 for h2,y2 in samples[i+1:] if abs(y2-y1) > 1e-12]
    if not gradients:
        raise ValueError('insufficient vertical span for repose profile')
    # Exhaustive pair hypotheses are deterministic for these small profiles.
    # Maximum consensus finds the supported face without an endpoint anchor.
    hypotheses = []
    for i, (h1,y1) in enumerate(samples):
        for h2,y2 in samples[i+1:]:
            if abs(y2-y1) <= 1e-12:
                continue
            b = (h2-h1)/(y2-y1)
            a = h1-b*y1
            distances = [abs(h-a-b*y)/math.hypot(1,b) for h,y in samples]
            inliers = [(error,w) for error,w in zip(distances,weights) if error <= limit]
            mass = sum(w for error,w in inliers)
            hypotheses.append((-round(mass,12), sum(w*error**2 for error,w in inliers)/mass, b,a))
    _,_,gradient,intercept_h = min(hypotheses)

    previous = None
    converged = False
    for _ in range(20):
        errors = [abs(h-intercept_h-gradient*y)/math.hypot(1,gradient) for h,y in samples]
        mask = [error <= limit for error in errors]
        retained = [p for p,keep in zip(samples,mask) if keep]
        retained_weights = [w for w,keep in zip(weights,mask) if keep]
        if len(retained) < 2:
            raise ValueError('no majority-supported repose line')
        mass = sum(retained_weights)
        mean_h = sum(w*h for (h,y),w in zip(retained,retained_weights))/mass
        mean_y = sum(w*y for (h,y),w in zip(retained,retained_weights))/mass
        denominator = sum(w*(y-mean_y)**2 for (h,y),w in zip(retained,retained_weights))
        if denominator <= 1e-20:
            raise ValueError('insufficient height variation for repose fit')
        gradient = sum(w*(y-mean_y)*(h-mean_h) for (h,y),w in zip(retained,retained_weights))/denominator
        intercept_h = mean_h-gradient*mean_y
        if mask == previous:
            converged = True
            break
        previous = mask
    if abs(gradient) <= 1e-12:
        slope = math.copysign(1e12, gradient or 1.0)
    else:
        slope = 1.0/gradient
    intercept = -intercept_h*slope
    errors = [abs(h-intercept_h-gradient*y)/math.hypot(1,gradient) for h,y in samples]
    downweighted = [point for point,error in zip(samples,errors) if error > limit]
    mask = [error <= limit for error in errors]
    inlier_points = [point for point,keep in zip(samples,mask) if keep]

    runs, current = [], []
    for index, keep in enumerate(mask):
        if not keep:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    # Distinct particles alone do not establish sustained geometry: several
    # sphere caps can switch within one radius of physical height (real trial26).
    # Both independent support AND physical extent are required. Endpoint runs
    # are retained explicitly; they are not mislabeled as middle departures.
    departure_runs = []
    for run in runs:
        distinct = len({ids[i] for i in run})
        span_radii = (samples[run[-1]][1]-samples[run[0]][1])/median_radius
        min_residual_radii = min(errors[i]/median_radius for i in run)
        signed_residuals = [samples[i][0]-intercept_h-gradient*samples[i][1] for i in run]
        same_side = min(signed_residuals)*max(signed_residuals) > 0
        # Test contiguous high-residual, same-sign subruns so an adjacent
        # moderate shoulder or opposite-sign outlier cannot mask a large step.
        large_step = False
        step = []
        for i, residual in zip(run, signed_residuals):
            if errors[i]/median_radius < 4.0-1e-10:
                step = []
                continue
            if step and residual*(samples[step[-1]][0]-intercept_h-gradient*samples[step[-1]][1]) <= 0:
                step = []
            step.append(i)
            if (len({ids[j] for j in step}) >= 3 and
                    (samples[step[-1]][1]-samples[step[0]][1])/median_radius >= 2.0-1e-10):
                large_step = True
                break
        departure_runs.append({
            'start_index': run[0], 'end_index': run[-1],
            'location': ('lower_endpoint' if run[0] == 0 else
                         'upper_endpoint' if run[-1] == len(samples)-1 else 'middle'),
            'distinct_particles': distinct, 'height_span_radii': span_radii,
            'minimum_orthogonal_residual_radii': min_residual_radii,
            'same_signed_departure': same_side, 'large_coherent_step': large_step,
            'sustained': (distinct >= 2 and span_radii >= 4.0-1e-10) or large_step,
        })
    sustained = any(run['sustained'] for run in departure_runs)
    height_gap = any(b[1]-a[1] > 4*median_radius+1e-10 for a,b in zip(samples,samples[1:]))
    mean_y = sum(w*y for (h,y),w in zip(samples,weights))/sum(weights)
    ss_tot = sum(w*(y-mean_y)**2 for (h,y),w in zip(samples,weights))
    ss_res = sum(w*(y-(slope*h+intercept))**2 for (h,y),w in zip(samples,weights))
    inlier_weights = [w for w,keep in zip(weights,mask) if keep]
    inlier_mean_y = sum(w*y for (h,y),w in zip(inlier_points,inlier_weights))/sum(inlier_weights)
    inlier_ss_tot = sum(w*(y-inlier_mean_y)**2 for (h,y),w in zip(inlier_points,inlier_weights))
    inlier_ss_res = sum(w*(y-(slope*h+intercept))**2 for (h,y),w in zip(inlier_points,inlier_weights))
    inlier_r2 = 1-inlier_ss_res/inlier_ss_tot if inlier_ss_tot else 0.0
    unique_support = len({index for index,keep in zip(ids,mask) if keep})
    # Keep the approved free-intercept h(y) refit unchanged. Geometry quality
    # uses distances to that same line, rather than inverting it and judging
    # y residuals (ill-conditioned for near-vertical profiles).
    inlier_mass = sum(inlier_weights)
    orthogonal_mse = sum(w*error*error for w,error,keep in zip(weights,errors,mask) if keep)/inlier_mass
    orthogonal_rms_radii = math.sqrt(orthogonal_mse)/median_radius
    tangent_coordinates = [(gradient*h+y)/math.hypot(1,gradient) for h,y in inlier_points]
    tangent_mean = sum(w*t for w,t in zip(inlier_weights,tangent_coordinates))/inlier_mass
    tangent_variance = sum(w*(t-tangent_mean)**2 for w,t in zip(inlier_weights,tangent_coordinates))/inlier_mass
    orthogonal_quality = 1-orthogonal_mse/tangent_variance if tangent_variance > 1e-20 else 0.0
    vertical_span_radii = (max(y for h,y in inlier_points)-min(y for h,y in inlier_points))/median_radius
    fitted_horizontal_span_radii = abs(gradient)*vertical_span_radii
    policy = {
        'version': 'repose-geometric-quality-v1',
        'r2_y_on_h_role': 'diagnostic_only',
        'minimum_unique_particles': 5,
        'minimum_vertical_span_radii': 4.0,
        'minimum_fitted_horizontal_span_radii': 2.0,
        'maximum_orthogonal_rms_radii': 1.0,
        'minimum_orthogonal_quality': 0.90,
        'maximum_unsupported_gap_radii': 4.0,
        'minimum_departure_distinct_particles': 2,
        'minimum_departure_height_span_radii': 4.0,
        'large_coherent_step_minimum_particles': 3,
        'large_coherent_step_minimum_height_span_radii': 2.0,
        'large_coherent_step_minimum_orthogonal_residual_radii': 4.0,
        'refit': 'free_intercept_weighted_horizontal_on_height_unchanged',
        'weighting': 'physical_height_trapezoidal_cells',
        'excluded_sample_weight': 0.0,
        'threshold_provenance': 'candidate engineering policy; not a universal physical standard',
    }
    reason = ('unstable surface consensus' if not converged else
              'insufficient unique particle support' if unique_support < 5 else
              'unsupported height gap in profile' if height_gap else
              'sustained unsupported profile departure' if sustained else
              'no majority-supported face' if inlier_mass <= sum(weights)/2 else
              'insufficient physical height support' if vertical_span_radii < 4.0 else
              'near-vertical profile lacks resolved horizontal retreat' if fitted_horizontal_span_radii < 2.0 else
              'orthogonal residual exceeds particle-scale quality limit' if orthogonal_rms_radii > 1.0 else
              'insufficient dimensionless orthogonal line quality' if orthogonal_quality < 0.90 else '')
    return {'slope':slope, 'intercept':intercept,
            'angle_deg':abs(math.degrees(math.atan(slope))),
            'r2':inlier_r2, 'all_samples_r2':1-ss_res/ss_tot if ss_tot else 0.0,
            'valid':not reason, 'rejection_reason':reason,
            'inlier_mask':mask, 'inlier_points':inlier_points,
            'downweighted_points':downweighted,
            'orthogonal_residuals':errors,
            'orthogonal_rms_radii': orthogonal_rms_radii,
            'orthogonal_quality': orthogonal_quality,
            'orthogonal_quality_definition': '1 - weighted perpendicular MSE / weighted tangential variance',
            'vertical_span_radii': vertical_span_radii,
            'fitted_horizontal_span_radii': fitted_horizontal_span_radii,
            'validity_policy': policy,
            'fit_sample_weights': [w if keep else 0.0 for w,keep in zip(weights,mask)],
            'excluded_sample_semantics': 'zero fit weight; downweighted_points is a legacy alias, not Huber',
            'departure_runs': departure_runs,
            'sustained_departure_policy': {'minimum_distinct_particles': 2, 'minimum_height_span_radii': 4.0},
            'horizontal_gradient':gradient, 'horizontal_intercept':intercept_h}
