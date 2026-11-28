"""Fit the intrinsic spectrum on benchtop coupons and predict the calibrated chamber response."""
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


def _bridge(cal):
    raw = np.asarray(cal['bridge_raw'])
    loads = np.asarray(cal['true_channel_loads'])
    return np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))), loads, rcond=None)[0].sum(axis=1)


def _loads(part, bridge):
    return tuple(np.asarray(part[key]) @ bridge[:-1] + bridge[-1]
                 for key in ('past_channel_raw', 'hold_channel_raw'))


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


# Initial-slope fractions identified on complete public chamber coupons.
_CHAMBER_FAST = 0.4744670978
_CHAMBER_LINEAR = -0.0292589413


def _chamber(taus, theta, hold, features, times):
    slow = 1.0 - _CHAMBER_FAST - _CHAMBER_LINEAR
    fractions = np.array([_CHAMBER_FAST, slow])
    kernel = -np.expm1(-times[None, :] / taus[:, None])
    score = theta[2] + features @ theta[3:]
    stress_profile = theta[:2] @ kernel
    feature_profile = (fractions * taus) @ kernel + _CHAMBER_LINEAR * times
    increments = hold[:, None] * stress_profile + score[:, None] * feature_profile
    rate = hold * (theta[:2] @ (1.0 / taus)) + score
    return increments, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    features = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, times = (np.asarray(dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train, bridge), features[:n], dt, times,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    _, hold = _loads(test, bridge)
    increments, rate = _chamber(taus, theta, hold, features[n:], times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
             'predicted_strain': np.asarray(test['strain_0'])[:, None] + increments,
             'initial_rate_pred': rate}
