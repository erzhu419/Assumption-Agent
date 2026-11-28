"""Fit a two-branch hereditary model to calibrated coupon measurements."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    labs = np.concatenate([train['lab_id'], test['lab_id']])
    strata = np.concatenate([train['stratum'], test['stratum']])
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, count = np.unique(np.column_stack([labs, strata]), axis=0,
                               return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell].reshape(-1, 1)
    present = ~np.isnan(x)
    mean = np.nansum(weights * x, axis=0) / np.sum(weights * present, axis=0)
    return np.column_stack([np.ones(len(x)), np.where(present, x, mean)])


def _bridge_basis(loads):
    prev = np.roll(loads, 1, axis=-1)
    nxt = np.roll(loads, -1, axis=-1)
    return np.stack((np.ones_like(loads), loads, prev, nxt, loads * nxt,
                     loads ** 3), axis=-1)


def _fit_bridge(calibration):
    raw = np.asarray(calibration['bridge_raw'])
    basis = _bridge_basis(np.asarray(calibration['true_channel_loads']))
    keep = np.ones(len(raw), dtype=bool)
    for _ in range(4):
        coef = np.array([np.linalg.lstsq(basis[keep, j], raw[keep, j], rcond=None)[0]
                         for j in range(4)])
        error = np.linalg.norm(np.sum(basis * coef, axis=-1) - raw, axis=-1)
        cutoff = max(0.005, 4 * np.median(error[keep]))
        updated = error < cutoff
        if np.array_equal(keep, updated):
            break
        keep = updated
    return coef


def _invert_bridge(raw, coef):
    shape = raw.shape
    raw = np.asarray(raw).reshape(-1, 4)
    j = np.arange(4)
    prev, nxt = (j - 1) % 4, (j + 1) % 4
    jac = np.zeros((len(raw), 4, 4))
    jac[:, j, prev] = coef[:, 2]
    jac[:, j, nxt] = coef[:, 3]
    jac[:, j, j] = coef[:, 1]
    loads = np.linalg.solve(jac, (raw - coef[:, 0])[..., None])[..., 0]
    for _ in range(20):
        err = np.sum(_bridge_basis(loads) * coef, axis=-1) - raw
        if np.max(np.abs(err)) < 1e-11:
            break
        jac[:, j, j] = coef[:, 1] + coef[:, 4] * loads[:, nxt] + 3 * coef[:, 5] * loads ** 2
        jac[:, j, nxt] = coef[:, 3] + coef[:, 4] * loads
        loads -= np.linalg.solve(jac, err[..., None])[..., 0]
    return loads.reshape(shape)


def _loads(coef, split):
    return tuple(_invert_bridge(split[key], coef).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(log_tau, past, hold, x, dt, times):
    inv_tau = np.exp(-np.asarray(log_tau))
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for duration, stress in zip(dt, past.T):
        u = duration * inv_tau
        a = np.exp(-u)
        dz = a * (dz + u * (z - stress[:, None]))
        z = a * z - np.expm1(-u) * stress[:, None]
    gap = hold[:, None] - z
    u = times[:, None] * inv_tau
    rise = -np.expm1(-u)
    curve = np.concatenate((gap[:, None, :] * rise, x[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack([gap * inv_tau, x])
    dcurve = -dz[:, None, :] * rise - gap[:, None, :] * u * np.exp(-u)
    drate = (-dz - gap) * inv_tau
    return curve, rate, dcurve, drate


def _fit(past, hold, x, dt, times, y, r):
    n_curve = y.size
    target = np.r_[y.ravel(), r]
    rate_unit = 1.0 / np.median(times)
    units = np.r_[np.ones(n_curve), np.full(len(r), rate_unit)]
    lower = np.r_[np.log(0.05 + 1e-8), np.log(0.05 + 1e-8), np.full(7, -np.inf)]
    upper = np.r_[np.log(25.0 - 1e-8), np.log(25.0 - 1e-8), np.full(7, np.inf)]

    def arrays(p):
        curve, rate, dc, dr = _design(p[:2], past, hold, x, dt, times)
        a = np.vstack([curve.reshape(-1, 7), rate])
        da = np.vstack([dc.reshape(-1, 2), dr])
        return a, da

    def fit(start, scale):
        def residual(p):
            a, _ = arrays(p)
            return (a @ p[2:] - target) / scale

        def jacobian(p):
            a, da = arrays(p)
            return np.column_stack([da * p[2:4], a]) / scale[:, None]

        return least_squares(residual, start, jac=jacobian, bounds=(lower, upper),
                             loss='soft_l1', x_scale='jac', max_nfev=250,
                             ftol=1e-10, gtol=1e-10, xtol=1e-10)

    candidates = []
    for tau in ((0.25, 1.5), (0.7, 4.0), (0.15, 10.0), (2.0, 15.0)):
        log_tau = np.log(tau)
        a, _ = arrays(log_tau)
        theta, *_ = np.linalg.lstsq(a / units[:, None], target / units, rcond=None)
        candidates.append(fit(np.r_[log_tau, theta], 0.02 * units))
    p = min(candidates, key=lambda candidate: candidate.cost).x
    for _ in range(2):
        a, _ = arrays(p)
        err = a @ p[2:] - target
        scales = []
        for group in (err[:n_curve], err[n_curve:]):
            scales.append(max(1e-7, 1.4826 * np.median(np.abs(group - np.median(group)))))
        p = fit(p, 1.5 * np.repeat(scales, [n_curve, len(r)])).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    coef = _fit_bridge(dataset['calibration_load_cells'])
    past, hold = _loads(coef, train)
    x = _features(train, test)
    tau, theta = _fit(past, hold, x[:len(hold)], dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    past_test, hold_test = _loads(coef, test)
    curve, rate, _, _ = _design(np.log(tau), past_test, hold_test, x[len(hold):], dt, times)
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
