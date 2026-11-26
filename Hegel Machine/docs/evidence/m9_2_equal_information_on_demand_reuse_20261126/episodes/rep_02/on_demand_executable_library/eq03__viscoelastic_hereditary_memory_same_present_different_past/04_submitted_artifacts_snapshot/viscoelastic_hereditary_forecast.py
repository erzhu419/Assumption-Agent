"""Coupon-calibrated two-mode hereditary creep and first-crossing forecasts."""

import numpy as np


def _align(train, test):
    def pool(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs, strata = pool('lab_id'), pool('stratum')
    raw = np.asarray(pool('coupon_features_raw'), dtype=float)
    reference = np.asarray(pool('ref_coupons'), dtype=float)
    aligned = np.empty_like(raw)
    for lab in np.unique(labs):
        rows = labs == lab
        controls = reference[rows]
        controls = controls[:, np.any(np.isfinite(controls), axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - offset) / gain

    _, inverse, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                   return_inverse=True, return_counts=True)
    weights = (len(labs) / (len(counts) * counts[inverse]))[:, None]
    finite = np.isfinite(aligned)
    fill = np.sum(weights * np.where(finite, aligned, 0.0), axis=0)
    mass = np.sum(weights * finite, axis=0)
    np.divide(fill, mass, out=fill, where=mass > 0)
    return np.where(finite, aligned, fill)


def _design(split, x, dt, taus, times):
    history = np.asarray(split['past_stress_history'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    state = np.zeros((len(x), len(taus)))
    for load in history.T:
        state = state * decay + load[:, None] * injection

    gaps = np.asarray(split['stress_hold_0'], dtype=float)[:, None] - state
    relaxation = -np.expm1(-times[:, None] / taus)
    rates = np.column_stack((np.ones(len(x)), x))
    return np.concatenate((gaps[:, None, :] * relaxation[None, :, :],
                           rates[:, None, :] * times[None, :, None]), axis=2)


def _cauchy_fit(design, observed):
    # Equilibrate columns so the fast and slow modes are not numerically
    # penalized by their smaller units relative to the linear creep terms.
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1.0
    a = design / norms
    beta = np.linalg.lstsq(a, observed, rcond=None)[0]
    floor = 1e-9 * max(1.0, np.median(np.abs(observed)))
    for _ in range(60):
        residual = observed - a @ beta
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        root_weight = 1.0 / np.hypot(1.0, residual / (1.5 * scale))
        next_beta = np.linalg.lstsq(a * root_weight[:, None],
                                     observed * root_weight, rcond=None)[0]
        converged = np.linalg.norm(next_beta - beta) <= 1e-10 * (1 + np.linalg.norm(beta))
        beta = next_beta
        if converged:
            break
    return beta / norms


def _crossings(times, initial, forecast, critical):
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = np.argmax(reached, axis=1)
    result = np.full(len(initial), times[-1])
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    end = first[rows]
    start = end - 1
    ya, yb = values[rows, start], values[rows, end]
    result[rows] = knots[start] + ((critical[rows] - ya) / (yb - ya)) * (knots[end] - knots[start])
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit on observed training curves and forecast every test coupon."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    cut = len(train['lab_id'])
    aligned = _align(train, test)
    x_test = aligned[cut:]
    design = _design(train, aligned[:cut], dt, taus, times)
    residual_target = (np.asarray(train['strain_curves_obs'], dtype=float)
                       - np.asarray(train['strain_0'], dtype=float)[:, None])
    coeff = _cauchy_fit(design.reshape(-1, design.shape[-1]), residual_target.ravel())
    initial = np.asarray(test['strain_0'], dtype=float)
    forecast = initial[:, None] + _design(test, x_test, dt, taus, times) @ coeff
    crossing = _crossings(times, initial, forecast, np.asarray(test['critical_strain'], dtype=float))
    return {'aligned_test_features': x_test, 'hereditary_weights': coeff,
            'predicted_strain': forecast, 'crossing_times': crossing}
