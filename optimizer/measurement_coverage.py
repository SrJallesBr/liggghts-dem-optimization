"""Honest paired-angle coverage: missing responses never count as successes."""

def summarize_coverage(rows, target_fraction=0.90):
    if not 0 < target_fraction <= 1:
        raise ValueError('coverage target must be in (0,1]')
    eligible=[r for r in rows if r['discharge']['available']]
    paired=[r for r in eligible if r['repose']['available']]
    valid=[r for r in paired if r['discharge']['valid'] and r['repose']['valid']]
    return {
        'target_fraction':target_fraction,
        'completed_discharge_trials':len(eligible),
        'discharge_valid_trials':sum(r['discharge']['valid'] for r in eligible),
        'repose_snapshots_available':sum(r['repose']['available'] for r in rows),
        'repose_valid_trials':sum(r['repose']['available'] and r['repose']['valid'] for r in rows),
        'paired_snapshots_available':len(paired),
        'paired_valid_trials':len(valid),
        'paired_coverage_all_completed_discharge_pct':100*len(valid)/len(eligible) if eligible else None,
        'paired_coverage_available_pairs_pct':100*len(valid)/len(paired) if paired else None,
        'missing_repose_trials':[r['trial'] for r in eligible if not r['repose']['available']],
        'target_reached':bool(eligible) and len(valid)/len(eligible)>=target_fraction,
    }
