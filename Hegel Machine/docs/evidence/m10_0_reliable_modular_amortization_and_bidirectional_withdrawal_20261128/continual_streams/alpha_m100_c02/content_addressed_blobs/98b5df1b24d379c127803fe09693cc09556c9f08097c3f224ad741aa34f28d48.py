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


def _stress(block):
    return tuple(np.asarray(block[key], dtype=float).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _intrinsic(dataset):
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[key], dtype=float) for key in ('past_dt', 'eval_times'))
    x = _features(train, test)
    past, hold = _stress(train)
    n = len(hold)
    tau, theta = _fit(past, hold, x[:n], dt, times,
                     train['strain_curves_obs'] - train['strain_0'][:, None],
                     train['instant_rate_obs'])
    return dt, times, x[n:], tau, theta


# Fractions of the initial drift slope carried by the two chamber modes,
# measured on the calibrated chamber verification coupons.
_CHAMBER_FRACTIONS = np.array([0.19683, 0.53764])


def _chamber_response(hold, x, strain_0, times, tau, theta, fractions):
    rise = -np.expm1(-times[:, None] / tau)
    drift = x @ theta[2:]
    # The chamber resets both past-load states. Each drift correction has
    # zero slope at the origin, so it reshapes the curve but not its rate.
    drift_kernel = times + (rise * tau - times[:, None]) @ fractions
    strain = strain_0[:, None] + hold[:, None] * (rise @ theta[:2])
    strain += drift[:, None] * drift_kernel
    rate = hold * np.sum(theta[:2] / tau) + drift
    return strain, rate


def solve_downstream(cal_item: dict) -> dict:
    dt, times, x, tau, theta = _intrinsic(cal_item)
    test = cal_item['test']
    _, hold = _stress(test)
    strain, rate = _chamber_response(hold, x, test['strain_0'], times, tau,
                                     theta, _CHAMBER_FRACTIONS)
    return dict(estimated_taus=tau, hereditary_weights=theta,
                predicted_strain=strain, initial_rate_pred=rate)
