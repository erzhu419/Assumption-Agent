"""Calibrate laboratory coupons and forecast creep from causal Prony states."""

import numpy as np


def _align(train, test):
    def pool(name):
        return np.concatenate((train[name], test[name]), axis=0)

    lab, stratum = pool('lab_id'), pool('stratum')
    controls = pool('ref_coupons')
    features = pool('coupon_features_raw').astype(float, copy=True)
    for group in np.unique(lab):
        rows = lab == group
        panel = controls[rows]
        valid = np.sum(np.isfinite(panel), axis=0) >= 2
        panel = panel[:, valid]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[rows] = (features[rows] - offset) / gain

    _, cells, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    weight = (len(lab) / (len(counts) * counts[cells.ravel()]))[:, None]
    finite = np.isfinite(features)
    mass = np.sum(weight * finite, axis=0)
    total = np.sum(weight * np.where(finite, features, 0.0), axis=0)
    fill = np.divide(total, mass, out=np.zeros_like(total), where=mass > 0)
    return np.where(finite, features, fill)


def _design(split, features, dt, taus, times):
    history = np.asarray(split['past_stress_history'], dtype=float)
    decay = np.exp(-dt / taus)
    update = -np.expm1(-dt / taus)
    state = np.zeros((history.shape[0], taus.size))
    for stress in history.T:
        state = state * decay + stress[:, None] * update

    gap = np.asarray(split['stress_hold_0'])[:, None] - state
    relaxation = -np.expm1(-times[:, None] / taus)
    design = np.empty((len(history), len(times), 7))
    design[:, :, :2] = gap[:, None, :] * relaxation[None, :, :]
    design[:, :, 2] = times
    design[:, :, 3:] = features[:, None, :] * times[None, :, None]
    return design


def _cauchy_fit(design, target):
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ theta
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = max(1.4826 * mad, 1e-3)
    for _ in range(60):
        residual = target - design @ theta
        root_weight = 1.0 / np.hypot(1.0, residual / scale)
        updated = np.linalg.lstsq(design * root_weight[:, None],
                                  target * root_weight, rcond=None)[0]
        if np.linalg.norm(updated - theta) <= 1e-10 * (1 + np.linalg.norm(updated)):
            return updated
        theta = updated
    return theta


def _first_crossing(times, initial, predicted, thresholds):
    knots = np.concatenate(([0.0], times))
    strain = np.column_stack((initial, predicted))
    reached = strain >= thresholds[:, None]
    first = np.argmax(reached, axis=1)
    result = np.full(initial.shape, times[-1], dtype=float)
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    lo, hi = strain[rows, j - 1], strain[rows, j]
    result[rows] = knots[j - 1] + ((thresholds[rows] - lo) / (hi - lo)) * (knots[j] - knots[j - 1])
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit seven hereditary coefficients from training curves and forecast the test hold."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    aligned = _align(train, test)
    n = len(train['lab_id'])
    train_design = _design(train, aligned[:n], dt, taus, times)
    target = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    theta = _cauchy_fit(train_design.reshape(-1, 7), target.ravel())

    initial = np.asarray(test['strain_0'], dtype=float)
    predicted = initial[:, None] + _design(test, aligned[n:], dt, taus, times) @ theta
    return {
        'aligned_test_features': aligned[n:],
        'hereditary_weights': theta,
        'predicted_strain': predicted,
        'crossing_times': _first_crossing(times, initial, predicted, np.asarray(test['critical_strain'])),
    }
