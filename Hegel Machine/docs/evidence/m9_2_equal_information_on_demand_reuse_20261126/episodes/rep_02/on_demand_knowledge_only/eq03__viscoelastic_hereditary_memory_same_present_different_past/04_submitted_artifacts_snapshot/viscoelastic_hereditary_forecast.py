"""Coupon-calibrated two-mode hereditary creep and threshold forecasts."""

import numpy as np


def _aligned_features(train, test):
    def pooled(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs = pooled('lab_id')
    strata = pooled('stratum')
    features = pooled('coupon_features_raw').astype(float, copy=False)
    controls = pooled('ref_coupons').astype(float, copy=False)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = controls[rows]
        counts = np.isfinite(panel).sum(axis=0)
        offset = np.median(np.nanmean(panel[:, counts > 0], axis=0))
        gain = np.median(np.nanstd(panel[:, counts >= 2], axis=0, ddof=0))
        features[rows] = (features[rows] - offset) / gain

    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = (len(labs) / (len(counts) * counts[cell]))[:, None]
    finite = np.isfinite(features)
    mass = np.sum(weights * finite, axis=0)
    sums = np.sum(weights * np.where(finite, features, 0.0), axis=0)
    means = np.divide(sums, mass, out=np.zeros_like(sums), where=mass > 0)
    features = np.where(finite, features, means)
    n_train = len(train['lab_id'])
    return features[:n_train], features[n_train:]


def _design(split, features, times, taus, dt):
    history = np.asarray(split['past_stress_history'], dtype=float)
    hold = np.asarray(split['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    state = np.zeros((len(hold), len(taus)))
    for stress in history.T:
        state = decay * state + injection * stress[:, None]
    growth = -np.expm1(-times[:, None] / taus)

    design = np.empty((len(hold), len(times), 3 + features.shape[1]))
    design[:, :, :2] = (hold[:, None] - state)[:, None, :] * growth[None, :, :]
    design[:, :, 2] = times
    design[:, :, 3:] = features[:, None, :] * times[None, :, None]
    return design


def _fit_cauchy(design, target):
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    # Refresh the residual scale after robust warm-up so initial spikes do not
    # set the bandwidth of the final Cauchy fit.
    for iterations in (8, 32):
        residual = target - design @ theta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(2.385 * 1.4826 * mad, 1e-8)
        for _ in range(iterations):
            root_weight = 1.0 / np.hypot(1.0, residual / scale)
            update = np.linalg.lstsq(design * root_weight[:, None],
                                     target * root_weight, rcond=None)[0]
            delta = np.linalg.norm(update - theta)
            theta = update
            if delta <= 1e-10 * (1.0 + np.linalg.norm(theta)):
                break
            residual = target - design @ theta
    return theta


def _crossing_times(times, initial, predicted, critical):
    crossing = np.full(len(initial), times[-1], dtype=float)
    at_start = initial >= critical
    crossing[at_start] = 0.0
    above = predicted >= critical[:, None]
    rows = np.flatnonzero(~at_start & above.any(axis=1))
    right = above[rows].argmax(axis=1)
    knots = np.r_[0.0, times]
    t_a, t_b = knots[right], knots[right + 1]
    y_a = np.column_stack((initial, predicted))[rows, right]
    y_b = predicted[rows, right]
    crossing[rows] = t_a + (critical[rows] - y_a) / (y_b - y_a) * (t_b - t_a)
    return crossing


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the causal Prony model on training curves and forecast test coupons."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    x_train, x_test = _aligned_features(train, test)
    design = _design(train, x_train, times, taus, dt)
    strain0 = np.asarray(train['strain_0'], dtype=float)
    target = np.asarray(train['strain_curves_obs'], dtype=float) - strain0[:, None]
    theta = _fit_cauchy(design.reshape(-1, design.shape[-1]), target.ravel())

    initial = np.asarray(test['strain_0'], dtype=float)
    predicted = initial[:, None] + _design(test, x_test, times, taus, dt) @ theta
    critical = np.asarray(test['critical_strain'], dtype=float)
    return {
        'aligned_test_features': x_test,
        'hereditary_weights': theta,
        'predicted_strain': predicted,
        'crossing_times': _crossing_times(times, initial, predicted, critical),
    }
