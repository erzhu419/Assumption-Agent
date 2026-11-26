"""Fit preloaded creep dynamics and read out secant compliance and crossings."""

import numpy as np
from scipy.optimize import least_squares


def _aligned_features(train, test):
    n = len(train['lab_id'])
    lab = np.r_[train['lab_id'], test['lab_id']]
    panel = np.r_[train['ref_coupons'], test['ref_coupons']]
    raw = np.r_[train['coupon_features_raw'], test['coupon_features_raw']]
    aligned = np.empty_like(raw, dtype=float)
    for group in np.unique(lab):
        rows = lab == group
        controls = panel[rows]
        center = np.nanmedian(np.nanmean(controls, axis=0))
        gain = np.nanmedian(np.nanstd(controls, axis=0))
        aligned[rows] = (raw[rows] - center) / gain
    missing = ~np.isfinite(aligned)
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, cell, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weight = 1.0 / count[cell.reshape(-1)]
    observed = ~missing[:n]
    total = weight @ observed
    fill = np.divide(weight @ np.where(observed, aligned[:n], 0.0), total,
                     out=np.zeros(raw.shape[1]), where=total > 0)
    features = np.column_stack((np.ones(len(raw)), np.where(missing, fill, aligned)))
    return features[:n], features[n:], ~missing[:n].any(axis=1), ~missing[n:].any(axis=1)


def _deficit(taus, dt, stress, initial, hold):
    state = initial.copy()
    decay = np.exp(-dt[:, None] / taus)
    uptake = -np.expm1(-dt[:, None] / taus)
    for a, b, pulse in zip(decay, uptake, stress.T):
        state = state * a + pulse[:, None] * b
    return hold[:, None] - state


def _fit(train, t, dt, features, complete):
    stress = train['past_stress'][complete]
    initial = train['initial_memory_states'][complete]
    hold = train['stress_hold_0'][complete]
    curves = train['strain_curves_obs'][complete] - train['strain_0'][complete, None]
    rate = train['instant_rate_obs'][complete]
    x = features[complete]
    time_scale = np.hypot(0.018, 0.016 * t)

    def physical(p):
        fast, gap = p[:2]
        return np.array([fast, fast + (9.0 - fast) * gap])

    def residual(p):
        taus = physical(p)
        amp = _deficit(taus, dt, stress, initial, hold) * p[2:4]
        flow = x @ p[4:]
        pred = amp @ (-np.expm1(-t[:, None] / taus)).T + flow[:, None] * t
        dr = amp @ (1.0 / taus) + flow - rate
        return np.r_[((pred - curves) / time_scale).ravel(), dr / 0.015]

    lower = np.r_[0.2, 1e-7, np.zeros(2), np.full(x.shape[1], -np.inf)]
    upper = np.r_[8.999, 1.0, np.full(2 + x.shape[1], np.inf)]
    fits = []
    for fast, slow in ((0.45, 3.0), (0.8, 6.0), (0.27, 1.5)):
        taus = np.array([fast, slow])
        q = _deficit(taus, dt, stress, initial, hold)
        init = np.linalg.lstsq(np.column_stack((q / taus, x)), rate, rcond=None)[0]
        init[:2] = np.maximum(init[:2], 0.05)
        p = np.r_[fast, (slow - fast) / (9.0 - fast), init]
        stage = least_squares(residual, p, bounds=(lower, upper), loss='soft_l1',
                              f_scale=2.5, max_nfev=120)
        fits.append(least_squares(residual, stage.x, bounds=(lower, upper),
                                  loss='cauchy', f_scale=1.5, max_nfev=120))
    p = min(fits, key=lambda fit: fit.cost).x
    return physical(p), p[2:4], p[4:]


def _crossing_times(t, initial, pred, critical):
    times = np.r_[0.0, t]
    curves = np.column_stack((initial, pred))
    reached = curves >= critical[:, None]
    first = np.argmax(reached, axis=1)
    crossing = np.full(len(initial), t[-1], dtype=float)
    crossing[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    left = curves[rows, j - 1]
    crossing[rows] = times[j - 1] + (times[j] - times[j - 1]) * (
        (critical[rows] - left) / (curves[rows, j] - left))
    return crossing


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    t, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    x_train, x_test, complete_train, complete_test = _aligned_features(train, test)
    taus, gains, drift = _fit(train, t, dt, x_train, complete_train)
    hold, initial = test['stress_hold_0'], test['strain_0']
    amp = _deficit(taus, dt, test['past_stress'], test['initial_memory_states'], hold) * gains
    measured_flow = test['instant_rate_obs'] - amp @ (1.0 / taus)
    coupon_flow = x_test @ drift
    coupon_weight = np.full(len(hold), 0.7)
    if np.any(~complete_test):
        # Downweight imputed drift according to the variance of the omitted terms.
        cells = np.column_stack((train['lab_id'][complete_train],
                                 train['stratum'][complete_train]))
        _, cell, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        weights = 1.0 / count[cell.reshape(-1)]
        missing = ~np.isfinite(test['coupon_features_raw'][~complete_test])
        omitted = (x_train[complete_train, 1:] * drift[1:]) @ missing.T
        center = np.average(omitted, axis=0, weights=weights)
        variance = np.average((omitted - center)**2, axis=0, weights=weights)
        coupon_weight[~complete_test] = 0.7 / (1.0 + 0.7 * variance / 0.012**2)
    flow = measured_flow + coupon_weight * (coupon_flow - measured_flow)
    pred = initial[:, None] + amp @ (-np.expm1(-t[:, None] / taus)).T + flow[:, None] * t
    return {'estimated_taus': taus, 'predicted_strain': pred,
            'secant_compliance': (pred[:, -1] - initial) / np.maximum(np.abs(hold), 0.5),
            'hazard_crossing_times': _crossing_times(t, initial, pred, test['critical_strain'])}
