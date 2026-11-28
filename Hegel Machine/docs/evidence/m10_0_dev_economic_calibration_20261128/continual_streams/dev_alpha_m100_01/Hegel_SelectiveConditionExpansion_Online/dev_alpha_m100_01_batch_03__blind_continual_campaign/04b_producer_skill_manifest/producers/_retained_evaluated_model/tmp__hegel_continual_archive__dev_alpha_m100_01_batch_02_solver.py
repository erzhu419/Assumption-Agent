"""Calibrated material fit and slope-preserving deployment chamber model."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    parts = (train, test)
    def pooled(key):
        return np.concatenate([np.asarray(p[key]) for p in parts], axis=0)

    labs = pooled('lab_id')
    strata = pooled('stratum')
    refs = pooled('ref_coupons')
    x = pooled('coupon_features_raw').astype(float, copy=True)
    for lab in np.unique(labs):
        m = labs == lab
        panel = refs[m]
        x[m] = (x[m] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = (1.0 / counts[cell])[:, None]
    valid = np.isfinite(x)
    means = np.sum(np.where(valid, x, 0.0) * w, axis=0) / np.sum(valid * w, axis=0)
    return np.where(valid, x, means)


def _design(taus, past, hold, x, dt, t):
    a = np.exp(-dt[:, None] / taus)
    b = -np.expm1(-dt[:, None] / taus)
    z = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        z = a[s] * z + b[s] * past[:, s, None]
    gap = hold[:, None] - z
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            t[None, :, None] * drift[:, None, :]), axis=2)
    return curve, np.column_stack((gap / taus, drift))


def _fit(past, hold, x, dt, t, curves, rates):
    yc, yr = curves.ravel(), rates.ravel()
    mc, mr = np.isfinite(yc), np.isfinite(yr)
    y = np.concatenate((yc[mc], yr[mr]))
    nc = mc.sum()

    def design(logtaus):
        c, r = _design(np.exp(logtaus), past, hold, x, dt, t)
        return np.vstack((c.reshape(-1, 7)[mc], r[mr]))

    def profile(logtaus):
        a = design(logtaus)
        return a @ np.linalg.lstsq(a, y, rcond=None)[0] - y

    lo, hi = np.log([0.050001, 24.9999])
    fits = [least_squares(profile, np.log(pair), bounds=(lo, hi), max_nfev=75)
            for pair in ((0.20, 1.4), (0.48, 3.8), (1.0, 10.0), (2.0, 18.0))]
    q = min(fits, key=lambda f: f.cost).x
    theta = np.linalg.lstsq(design(q), y, rcond=None)[0]
    p = np.r_[q, theta]
    for _ in range(3):
        err = design(p[:2]) @ p[2:] - y
        def scale(a):
            return max(1e-8, 1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else 1.0
        scales = np.r_[np.full(nc, scale(err[:nc])), np.full(len(y)-nc, scale(err[nc:]))]
        p = least_squares(lambda v: (design(v[:2]) @ v[2:] - y) / scales,
                          p, bounds=(np.r_[lo, lo, np.full(7, -np.inf)],
                                     np.r_[hi, hi, np.full(7, np.inf)]),
                          loss='soft_l1', f_scale=1.5, max_nfev=120).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


# Diagnostic leave-one-lab-out selects a two-mode, slope-preserving
# redistribution of the coupon drift score. Load memory is unchanged.
_CHAMBER_REDISTRIBUTION = np.array([0.5731, 0.7722])


def _loads(part):
    return tuple(np.asarray(part[k]).sum(axis=-1)
                 for k in ('past_channel_raw', 'hold_channel_raw'))


def _chamber(taus, theta, past, hold, x, dt, times):
    curve, rate = _design(taus, past, hold, x, dt, times)
    score = theta[2] + x @ theta[3:]
    zero_slope = taus * -np.expm1(-times[:, None] / taus) - times[:, None]
    return curve @ theta + score[:, None] * (zero_slope @ _CHAMBER_REDISTRIBUTION), rate @ theta


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    n = len(train['lab_id'])
    x = _features(train, test)
    dt, times = (np.asarray(cal_item[k]) for k in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train), x[:n], dt, times,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    increments, rates = _chamber(taus, theta, *_loads(test), x[n:], dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + increments,
            'initial_rate_pred': rates}
