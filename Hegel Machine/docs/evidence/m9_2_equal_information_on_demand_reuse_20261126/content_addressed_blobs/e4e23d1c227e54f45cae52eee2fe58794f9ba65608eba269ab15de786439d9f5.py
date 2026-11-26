"""Coupon-calibrated creep forecasts with two causal Prony memory modes."""

import numpy as np


def _aligned_features(train, test):
    def pooled(key, dtype=None):
        return np.concatenate((np.asarray(train[key], dtype=dtype),
                               np.asarray(test[key], dtype=dtype)))

    labs = pooled('lab_id')
    strata = pooled('stratum')
    controls = pooled('ref_coupons', float)
    aligned = pooled('coupon_features_raw', float)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = controls[rows]
        observed = np.isfinite(panel)
        valid = observed.sum(axis=0) >= 2
        panel = np.where(observed[:, valid], panel[:, valid], np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - offset) / gain

    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = labs.size / (counts.size * counts[cell.ravel()])
    finite = np.isfinite(aligned)
    support = weights @ finite
    means = np.divide(weights @ np.where(finite, aligned, 0.0), support,
                      out=np.zeros(aligned.shape[1]), where=support > 0)
    return np.where(finite, aligned, means)


def _design(coupons, features, times, taus, dt):
    stress = np.asarray(coupons['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    step = -np.expm1(-dt / taus)
    memory = np.zeros((stress.size, taus.size))
    for load in np.asarray(coupons['past_stress_history'], dtype=float).T:
        memory = memory * decay + load[:, None] * step
    relaxation = -np.expm1(-times[:, None] / taus)
    drift = np.column_stack((np.ones(stress.size), features))
    return np.concatenate(((stress[:, None] - memory)[:, None, :] * relaxation,
                           drift[:, None, :] * times[None, :, None]), axis=2)


def _cauchy_fit(design, target):
    def scale_from(residual):
        mad = np.median(np.abs(residual - np.median(residual)))
        return max(1e-3, 2.0 * 1.4826 * mad)

    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ beta
    scale = scale_from(residual)
    # Re-estimate scale once after the pilot so large spikes cannot set the
    # final loss width through their influence on the initial least squares.
    for iterations in (8, 25):
        for _ in range(iterations):
            root_weight = scale / np.hypot(scale, residual)
            updated = np.linalg.lstsq(design * root_weight[:, None],
                                      target * root_weight, rcond=None)[0]
            change = np.linalg.norm(updated - beta)
            beta = updated
            residual = target - design @ beta
            if change <= 1e-9 * (1.0 + np.linalg.norm(beta)):
                break
        scale = scale_from(residual)
    return beta


def _first_crossing(times, initial, forecast, critical):
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = np.argmax(reached, axis=1)
    result = np.full(initial.size, times[-1])
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    right = first[rows]
    left = right - 1
    before = values[rows, left]
    after = values[rows, right]
    result[rows] = (knots[left] + (critical[rows] - before) / (after - before)
                    * (knots[right] - knots[left]))
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the seven hereditary coefficients and forecast test coupon hazards."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    initial_train = np.asarray(train['strain_0'], dtype=float)
    initial_test = np.asarray(test['strain_0'], dtype=float)
    aligned = _aligned_features(train, test)
    test_features = aligned[initial_train.size:]
    train_design = _design(train, aligned[:initial_train.size], times, taus, dt)
    target = np.asarray(train['strain_curves_obs'], dtype=float) - initial_train[:, None]
    beta = _cauchy_fit(train_design.reshape(-1, train_design.shape[-1]), target.ravel())
    forecast = initial_test[:, None] + _design(test, test_features, times, taus, dt) @ beta
    critical = np.asarray(test['critical_strain'], dtype=float)
    return {'aligned_test_features': test_features,
            'hereditary_weights': beta,
            'predicted_strain': forecast,
            'crossing_times': _first_crossing(times, initial_test, forecast, critical)}
