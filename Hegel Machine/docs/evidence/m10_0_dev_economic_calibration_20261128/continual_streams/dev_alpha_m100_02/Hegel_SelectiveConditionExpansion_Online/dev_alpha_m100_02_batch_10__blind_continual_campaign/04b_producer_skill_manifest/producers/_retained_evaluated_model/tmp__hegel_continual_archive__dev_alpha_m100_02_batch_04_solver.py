"""Intrinsic relaxation fit in calibrated load coordinates."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test, extra=None):
    parts = (train, test)
    keys = ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')
    x, ref, lab, cell = (np.concatenate([p[k] for p in parts]) for k in keys)
    for j in np.unique(lab):
        r = ref[lab == j]
        x[lab == j] = (x[lab == j] - np.median(r.mean(axis=0))) / np.median(r.std(axis=0))
    _, ci, counts = np.unique(np.column_stack((lab, cell)), axis=0, return_inverse=True, return_counts=True)
    w = 1 / counts[ci]
    good = np.isfinite(x)
    mu = np.nansum(w[:, None] * x, axis=0) / (w @ good)
    x = np.where(good, x, mu)
    n = len(train['lab_id'])
    if extra is None:
        return x[:n], x[n:]
    z = extra['coupon_features_raw'].copy()
    for j in np.unique(extra['lab_id']):
        r = ref[lab == j]
        loc = extra['lab_id'] == j
        z[loc] = (z[loc] - np.median(r.mean(axis=0))) / np.median(r.std(axis=0))
    return x[:n], x[n:], np.where(np.isfinite(z), z, mu)


def _design(part, x, dt, t, tau):
    tau = np.asarray(tau, dtype=float)
    p = part['past_channel_raw'].sum(axis=-1)
    h = part['hold_channel_raw'].sum(axis=-1)
    u = dt[:, None] / tau
    future = np.cumsum(u[::-1], axis=0)[::-1] - u
    gap = h[:, None] - p @ (-np.expm1(-u) * np.exp(-future))
    drift = np.column_stack((np.ones(len(x)), x))
    time = np.r_[t, 1.]
    relax = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    return np.concatenate((gap[:, None, :] * relax, drift[:, None, :] * time[None, :, None]), axis=-1)


def _response(part):
    return np.column_stack((part['strain_curves_obs'] - part['strain_0'][:, None], part['instant_rate_obs']))


def _fit(part, x, dt, t):
    y = _response(part).ravel()
    def matrix(tau):
        return _design(part, x, dt, t, tau).reshape(-1, 7)
    def residual(p):
        return matrix(p[:2]) @ p[2:] - y
    def jac(p):
        a = matrix(p[:2]); cols = []
        for k in range(2):
            tm = p[:2].copy(); tp = tm.copy(); step = 1e-5 * tm[k]
            tm[k] -= step; tp[k] += step
            cols.append((matrix(tp)[:, k] - matrix(tm)[:, k]) * (p[2+k] / (2*step)))
        return np.column_stack((*cols, a))
    bounds = (np.r_[[.050001]*2, [-np.inf]*7], np.r_[[24.99999]*2, [np.inf]*7])
    best = None
    for tau in ([.4, 2.5], [.18, 6.], [.8, 12.]):
        beta = np.linalg.lstsq(matrix(tau), y, rcond=None)[0]
        f = least_squares(residual, np.r_[tau, beta], jac=jac, bounds=bounds, loss='cauchy', f_scale=.022, x_scale='jac', max_nfev=160)
        if best is None or f.cost < best.cost:
            best = f
    tau, theta = best.x[:2].copy(), best.x[2:].copy()
    if tau[0] > tau[1]:
        tau = tau[::-1].copy(); theta[:2] = theta[1::-1]
    return tau, theta


# Estimated from complete chamber verification coupons; the steady term
# carries the remainder so the initial drift rate is conserved.
_CHAMBER_TRANSIENT_FRACTIONS = np.array([0.19431366, 0.52714783])


def _chamber_response(part, x, t, tau, theta):
    hold = part['hold_channel_raw'].sum(axis=-1)
    drift = theta[2] + x @ theta[3:]
    f = _CHAMBER_TRANSIENT_FRACTIONS
    growth = -np.expm1(-t[:, None] / tau)
    amplitude = hold[:, None] * theta[:2] + drift[:, None] * (f * tau)
    strain = amplitude @ growth.T + (1 - f.sum()) * drift[:, None] * t
    rate = hold * np.sum(theta[:2] / tau) + drift
    return strain, rate


def solve_downstream(cal_item: dict) -> dict:
    tr, te = cal_item['train'], cal_item['test']
    dt, t = map(np.asarray, (cal_item['past_dt'], cal_item['eval_times']))
    xt, xe = _features(tr, te)
    tau, theta = _fit(tr, xt, dt, t)
    strain, rate = _chamber_response(te, xe, t, tau, theta)
    return dict(estimated_taus=tau, hereditary_weights=theta,
                predicted_strain=strain + te['strain_0'][:, None],
                initial_rate_pred=rate)
