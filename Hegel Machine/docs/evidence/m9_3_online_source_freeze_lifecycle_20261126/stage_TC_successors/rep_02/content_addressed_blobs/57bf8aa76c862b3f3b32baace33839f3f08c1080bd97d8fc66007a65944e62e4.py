"""Fit two preloaded hereditary branches and predict hold-phase readouts."""

import numpy as np
from scipy.optimize import least_squares


def _reference_location_scale(panel):
    available = np.isfinite(panel).sum(axis=0) >= 2
    panel = panel[:, available]
    return np.nanmedian(np.nanmean(panel, axis=0)), np.nanmedian(np.nanstd(panel, axis=0))


def _coupon_design(train, test):
    calibration = {lab: _reference_location_scale(train['ref_coupons'][train['lab_id'] == lab])
                   for lab in np.unique(train['lab_id'])}

    def align(split):
        x = np.empty_like(split['coupon_features_raw'], dtype=float)
        for lab in np.unique(split['lab_id']):
            rows = split['lab_id'] == lab
            if lab in calibration:
                location, scale = calibration[lab]
            else:
                location, scale = _reference_location_scale(split['ref_coupons'][rows])
            x[rows] = (split['coupon_features_raw'][rows] - location) / scale
        return x

    x, v = align(train), align(test)
    _, cell, count = np.unique(np.column_stack((train['lab_id'], train['stratum'])),
                               axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell]
    finite = np.isfinite(x)
    fill = (weights @ np.where(finite, x, 0.0)) / (weights @ finite)

    def design(raw):
        valid = np.isfinite(raw)
        return np.column_stack((np.ones(len(raw)), np.where(valid, raw, fill))), valid.all(axis=1)

    return design(x), design(v)


def _branches(split, dt, times, taus):
    z = np.asarray(split['initial_memory_states'], dtype=float).copy()
    increments = -np.expm1(-dt[:, None] / taus)
    for load, fraction in zip(split['past_stress'].T, increments):
        z += fraction * (load[:, None] - z)
    gap = split['stress_hold_0'][:, None] - z
    hold = -np.expm1(-times[:, None] / taus)
    return np.concatenate(((gap / taus)[:, None, :], gap[:, None, :] * hold), axis=1)


def _calibrate_drift(labs, coupon_drift, rate_drift, complete):
    # Reference location and scale errors act affinely on the linear coupon score.
    # Estimate the shared correction from paired measurements, not row-wise rates.
    result = rate_drift.copy()
    for lab in np.unique(labs):
        rows = (labs == lab) & complete
        score = coupon_drift[rows]
        if score.size == 0:
            continue
        centered = score - np.median(score)
        design = np.ones((score.size, 1))
        if np.ptp(score) > 0.0:
            design = np.column_stack((design, centered))
        discrepancy = rate_drift[rows] - score
        correction = least_squares(lambda c: design @ c - discrepancy,
                                   np.zeros(design.shape[1]), loss='soft_l1',
                                   f_scale=0.012).x
        result[rows] = score + design @ correction
    return result


def _first_crossings(initial, strain, critical, times):
    grid = np.r_[0.0, times]
    path = np.column_stack((initial, strain))
    reached = path >= critical[:, None]
    crossing = np.full(len(initial), times[-1], dtype=float)
    crossing[reached[:, 0]] = 0.0
    rows = np.flatnonzero(reached.any(axis=1) & ~reached[:, 0])
    right = reached[rows].argmax(axis=1)
    left = right - 1
    a, b = path[rows, left], path[rows, right]
    crossing[rows] = grid[left] + (grid[right] - grid[left]) * ((critical[rows] - a) / (b - a))
    return crossing


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    """Estimate shared relaxation times from training strains and initial rates."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    (x, complete), (v, test_complete) = _coupon_design(train, test)
    # Column zero models the physical derivative; all others model strain increments.
    time_basis = np.r_[1.0, times]
    drift_basis = x[:, None, :] * time_basis[None, :, None]
    target = np.column_stack((train['instant_rate_obs'],
                              train['strain_curves_obs'] - train['strain_0'][:, None]))
    weights = np.where(complete, 1.0, 0.35)[:, None]
    separation = 1e-4

    def unpack(p):
        fast, fraction = p[:2]
        return np.array([fast, fast + separation + fraction * (9.0 - fast - separation)])

    def residual(p):
        branch = _branches(train, dt, times, unpack(p)) @ p[2:4]
        drift = (x @ p[4:])[:, None] * time_basis
        return ((branch + drift - target) * weights).ravel()

    start = np.array([0.55, (3.0 - 0.55 - separation) / (9.0 - 0.55 - separation)])
    basis = np.concatenate((_branches(train, dt, times, unpack(start)), drift_basis), axis=2)
    coefficients = np.linalg.lstsq((basis * weights[:, :, None]).reshape(-1, basis.shape[-1]),
                                    (target * weights).ravel(), rcond=None)[0]
    p = least_squares(residual, np.r_[start, coefficients],
                      bounds=(np.r_[0.20, 0.0, np.full(len(coefficients), -np.inf)],
                              np.r_[9.0 - separation, 1.0, np.full(len(coefficients), np.inf)]),
                      loss='cauchy', f_scale=0.025, max_nfev=100).x
    taus = unpack(p)
    branch = _branches(test, dt, times, taus) @ p[2:4]
    # When a coupon is missing, the measured derivative identifies the net drift.
    drift = _calibrate_drift(test['lab_id'], v @ p[4:],
                             test['instant_rate_obs'] - branch[:, 0], test_complete)
    initial = test['strain_0']
    strain = initial[:, None] + branch[:, 1:] + drift[:, None] * times
    return {
        'estimated_taus': taus,
        'predicted_strain': strain,
        'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), 0.5),
        'hazard_crossing_times': _first_crossings(initial, strain, test['critical_strain'], times),
    }
