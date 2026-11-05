"""Offline response convention authorized by Henrique; not an angle estimator.

Preserves raw results. A nonempty, successfully processed snapshot rejected by
its estimator receives an adopted 90 deg value, explicitly USER ASSIGNED.
This rule does not detect vertical faces or prove cohesion as the cause.
Never apply it to missing, empty, corrupt or errored simulation artifacts.
No controller imports or campaign writes are performed here.
"""
import math

POLICY_VERSION = 'user-unmeasurable-as-90-v1'


def classify_retained_column(points, radii, *, center_xz, tube_radius, floor):
    """Conservative local final-snapshot geometry gate, not a cohesion diagnosis.

    A supported tall bulk must remain confined to the initial tube footprint
    and keep its robust radial width through three height bands. Thresholds
    are local provisional heuristics; no temporal or vendor equivalence claim.
    """
    import numpy as np
    p, r = np.asarray(points, float), np.asarray(radii, float)
    if p.ndim != 2 or p.shape[1] != 3 or r.shape != (len(p),):
        raise ValueError('finite XYZ and one positive radius per particle required')
    if (not np.isfinite(p).all() or not np.isfinite(r).all() or (r <= 0).any()
            or not np.isfinite(center_xz).all() or len(center_xz) != 2
            or not math.isfinite(tube_radius) or tube_radius <= 0 or not math.isfinite(floor)):
        raise ValueError('invalid particle or tube geometry')
    limits = {'minimum_particles': 100, 'minimum_band_particles': 30,
              'minimum_height_diameters': 8., 'minimum_confined_fraction': .90,
              'minimum_upper_lower_ratio': .85, 'maximum_upper_lower_ratio': 1.20,
              'maximum_band_relative_spread': .20, 'minimum_height_to_bulk_diameter': .75,
              'minimum_bulk_to_tube_radius': .80}
    out = {'version': 'retained-column-snapshot-local-v1', 'detected': False,
           'classification': 'inconclusive', 'thresholds': limits,
           'cohesion_cause_verified': False, 'temporal_no_spreading_verified': False,
           'equivalence_to_ansys_verified': False, 'tube_radius_m': float(tube_radius),
           'scope': 'final snapshot above tray; not a contact-network or motion-history test'}
    mask = p[:, 1] > floor
    p, r = p[mask], r[mask]
    out['particles_above_floor'] = len(p)
    if len(p) < limits['minimum_particles']:
        out['reason'] = 'insufficient bulk support'
        return out
    diameter = float(2*r.mean())
    low, high = np.quantile(p[:, 1], [.05, .95])
    height = float(high-low)
    radial = np.hypot(p[:, 0]-center_xz[0], p[:, 2]-center_xz[1])
    fraction = float(np.mean(radial <= tube_radius + diameter/2))
    out.update(height_span_m=height, height_span_diameters=height/diameter,
               confined_fraction=fraction, mean_diameter_m=diameter)
    band_radii, counts = [], []
    for lo, hi in [(0., .3), (.3, .6), (.6, .9)]:
        band = (p[:, 1] >= low+lo*height) & (p[:, 1] < low+hi*height)
        counts.append(int(band.sum()))
        if counts[-1] < limits['minimum_band_particles']:
            out.update(reason='insufficient height-band support', band_particle_counts=counts)
            return out
        band_radii.append(float(np.quantile(radial[band], .90)))
    if min(band_radii) <= 0:
        out['reason'] = 'degenerate radial bulk'
        return out
    ratio = band_radii[2]/band_radii[0]
    spread = (max(band_radii)-min(band_radii))/max(band_radii)
    out.update(band_r90_m=band_radii, band_particle_counts=counts,
               upper_lower_radius_ratio=ratio, band_relative_spread=spread,
               height_to_bulk_diameter=height/(2*max(band_radii)),
               bulk_to_tube_radius=min(band_radii)/tube_radius)
    detected = (height/diameter >= limits['minimum_height_diameters']
                and fraction >= limits['minimum_confined_fraction']
                and limits['minimum_upper_lower_ratio'] <= ratio <= limits['maximum_upper_lower_ratio']
                and spread <= limits['maximum_band_relative_spread']
                and out['height_to_bulk_diameter'] >= limits['minimum_height_to_bulk_diameter']
                and out['bulk_to_tube_radius'] >= limits['minimum_bulk_to_tube_radius'])
    out.update(detected=bool(detected), classification='retained_column' if detected else 'not_retained_column',
               reason='tall confined bulk with nearly constant radius over height' if detected else 'column geometry criteria not all met')
    return out


def apply_response_policy(row):
    result = dict(row)
    result.update(response_policy=POLICY_VERSION, adopted_angle_deg=None,
                  adopted_status='unresolved', measurement_valid=False,
                  assignment_reason=None, cohesion_cause_verified=False)
    if row.get('status') != 'processed' or row.get('particles', 0) <= 0:
        return result
    angle = row.get('new_angle_deg')
    if row.get('new_status') == 'measured' and isinstance(angle, (float, int)) and math.isfinite(angle):
        result.update(adopted_angle_deg=angle, adopted_status='measured', measurement_valid=True)
    elif row.get('new_status') == 'unmeasurable' and angle is None:
        diagnosis = row.get('geometry_diagnosis') or {}
        reason = ('Retained column geometry in final snapshot; adopted 90 deg, not a measured angle or verified cohesion cause.'
                  if diagnosis.get('detected') and diagnosis.get('classification') == 'retained_column'
                  else 'User convention for a nonempty snapshot with no accepted face fit; not a measured angle or verified cohesion diagnosis.')
        result.update(adopted_angle_deg=90.0, adopted_status='user_assigned_90',
                      assignment_reason=reason)
    return result
