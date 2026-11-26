"""Coupon calibration, hereditary creep forecasting, and first passage times."""

import numpy as np


def _align(train, test):
    def pooled(name, dtype):
        return np.concatenate((np.asarray(train[name], dtype=dtype),
                               np.asarray(test[name], dtype=dtype)), axis=0)

    lab = pooled('lab_id', int)
    stratum = pooled('stratum', int)
    ref = pooled('ref_coupons', float)
    aligned = pooled('coupon_features_raw', float)
    for lab_id in np.unique(lab):
        rows = lab == lab_id
        controls = ref[rows]
        controls = controls[:, np.any(np.isfinite(controls), axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - offset) / gain

    _, cell, count = np.unique(2 * lab + stratum, return_inverse=True,
                               return_counts=True)
    weights = (lab.size / count.size) / count[cell]
    valid = np.isfinite(aligned)
    weighted_mean = ((weights @ np.where(valid, aligned, 0.0)) /
                     (weights @ valid))
    return np.where(valid, aligned, weighted_mean)


def _design(split, aligned, dt, taus, times):
    history = np.asarray(split['past_stress_history'], dtype=float)
    decay = np.exp(-dt / taus)
    uptake = -np.expm1(-dt / taus)
    memory = np.zeros((history.shape[0], taus.size))
    for stress in history.T:
        memory = decay * memory + uptake * stress[:, None]

    hold = np.asarray(split['stress_hold_0'], dtype=float)
    response = -np.expm1(-times[:, None] / taus)
    transient = (hold[:, None] - memory)[:, None, :] * response[None, :, :]
    flow = np.column_stack((np.ones(len(aligned)), aligned))
    return np.concatenate((transient, times[None, :, None] * flow[:, None, :]),
                          axis=2)


def _fit(design, observations):
    theta = np.linalg.lstsq(design, observations, rcond=None)[0]
    # Re-estimate the inlier scale from the residual MAD; the floor also
    # allows IRLS to finish on data with exactly noiseless inliers.
    floor = 1e-8 * max(1.0, np.median(np.abs(observations)))
    for _ in range(80):
        residual = observations - design @ theta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(floor, 2.385 * 1.4826 * mad)
        sqrt_weight = scale / np.hypot(scale, residual)
        update = np.linalg.lstsq(design * sqrt_weight[:, None],
                                 observations * sqrt_weight, rcond=None)[0]
        if np.linalg.norm(update - theta) <= 1e-10 * max(1.0, np.linalg.norm(update)):
            return update
        theta = update
    return theta


def _first_passage(initial, strain, threshold, times):
    crossing = np.full(initial.shape, times[-1], dtype=float)
    at_start = initial >= threshold
    crossing[at_start] = 0.0
    reached = strain >= threshold[:, None]
    rows = np.flatnonzero(~at_start & reached.any(axis=1))
    knots = np.column_stack((initial, strain))
    time_knots = np.r_[0.0, times]
    left = reached[rows].argmax(axis=1)
    y_a, y_b = knots[rows, left], knots[rows, left + 1]
    t_a, t_b = time_knots[left], time_knots[left + 1]
    crossing[rows] = t_a + (threshold[rows] - y_a) / (y_b - y_a) * (t_b - t_a)
    return crossing


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the two-mode hereditary model and forecast test creep and crossings."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    initial_train = np.asarray(train['strain_0'], dtype=float)
    initial_test = np.asarray(test['strain_0'], dtype=float)
    n_train = initial_train.size
    aligned = _align(train, test)
    train_design = _design(train, aligned[:n_train], dt, taus, times)
    observations = np.asarray(train['strain_curves_obs'], dtype=float)
    theta = _fit(train_design.reshape(-1, train_design.shape[-1]),
                 (observations - initial_train[:, None]).ravel())
    aligned_test = aligned[n_train:]
    test_design = _design(test, aligned_test, dt, taus, times)
    strain = initial_test[:, None] + test_design @ theta
    threshold = np.asarray(test['critical_strain'], dtype=float)
    return {'aligned_test_features': aligned_test,
            'hereditary_weights': theta,
            'predicted_strain': strain,
            'crossing_times': _first_passage(initial_test, strain, threshold, times)}
