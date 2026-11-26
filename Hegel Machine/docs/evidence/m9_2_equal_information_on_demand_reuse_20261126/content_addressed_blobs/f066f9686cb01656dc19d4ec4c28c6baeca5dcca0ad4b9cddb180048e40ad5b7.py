"""Coupon-calibrated hereditary creep and first-passage forecasts."""

import numpy as np


def _align(train, test):
    """Estimate lab calibration and balanced imputation on the pooled coupons."""
    raw, refs, lab, stratum = (
        np.concatenate((train[key], test[key]), axis=0)
        for key in ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')
    )
    aligned = np.empty_like(raw, dtype=float)
    for label in np.unique(lab):
        mask = lab == label
        controls = refs[mask]
        controls = controls[:, np.isfinite(controls).any(axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - offset) / gain

    cells = np.column_stack((lab, stratum))
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = len(lab) / (len(counts) * counts[inverse])
    observed = np.isfinite(aligned)
    means = (weights @ np.where(observed, aligned, 0.0)) / (weights @ observed)
    aligned = np.where(observed, aligned, means)
    return np.split(aligned, [len(train['lab_id'])], axis=0)


def _design(split, features, times, dt, taus):
    """Build the linear basis for strain increments above the initial strain."""
    history = np.asarray(split['past_stress_history'], dtype=float)
    hold = np.asarray(split['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    memory = np.zeros((len(hold), len(taus)))
    for stress in history.T:
        memory = decay * memory + stress[:, None] * injection

    relaxation = -np.expm1(-times[:, None] / taus)
    design = np.empty((len(hold), len(times), len(taus) + 1 + features.shape[1]))
    design[:, :, :len(taus)] = (hold[:, None] - memory)[:, None, :] * relaxation
    design[:, :, len(taus)] = times
    design[:, :, len(taus) + 1:] = features[:, None, :] * times[None, :, None]
    return design


def _cauchy_fit(design, target):
    """Robust linear calibration by Cauchy IRLS with a MAD residual scale."""
    x = design.reshape(-1, design.shape[-1])
    y = target.ravel()
    # Equilibrate columns without changing the physical parameterization.
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1.0
    x = x / norm
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = 1e-12 * max(1.0, np.median(np.abs(y)))
    for _ in range(60):
        residual = y - x @ beta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(2.0 * 1.4826 * mad, floor)
        root_weight = scale / np.hypot(scale, residual)
        updated = np.linalg.lstsq(x * root_weight[:, None], y * root_weight, rcond=None)[0]
        if np.linalg.norm(updated - beta) <= 1e-10 * (1.0 + np.linalg.norm(updated)):
            beta = updated
            break
        beta = updated
    return beta / norm


def _first_crossing(initial, forecast, critical, times):
    """Interpolate the first upward crossing, including the initial knot."""
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = reached.argmax(axis=1)
    result = np.full(len(initial), times[-1], dtype=float)
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    right = first[rows]
    left = right - 1
    ya = values[rows, left]
    yb = values[rows, right]
    result[rows] = knots[left] + (critical[rows] - ya) / (yb - ya) * (knots[right] - knots[left])
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the two memory modes and predict test curves and crossing times."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    aligned_train, aligned_test = _align(train, test)
    design = _design(train, aligned_train, times, dt, taus)
    target = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    theta = _cauchy_fit(design, target)
    initial = np.asarray(test['strain_0'], dtype=float)
    forecast = initial[:, None] + _design(test, aligned_test, times, dt, taus) @ theta
    return {
        'aligned_test_features': aligned_test,
        'hereditary_weights': theta,
        'predicted_strain': forecast,
        'crossing_times': _first_crossing(initial, forecast, np.asarray(test['critical_strain']), times),
    }
