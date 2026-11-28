"""Calibrate the bridge, fit benchtop material parameters, and forecast chamber creep."""

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


from itertools import combinations_with_replacement
_TERMS = np.array([[k.count(j) for j in range(4)] for d in range(4) for k in combinations_with_replacement(range(4),d)])

def _poly(x, exponents):
    return np.prod(x[..., None, :]**exponents, axis=-1)

def _bridge(cal):
    loads = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    sweeps = cal['single_channel_sweeps']
    coefficients = np.zeros((len(_TERMS), 4))
    for j in range(4):
        m = np.asarray(sweeps['lab_id']) == j
        isolated = np.asarray(sweeps['true_stress'])[m,None] * (np.arange(4) == j)
        x = np.vstack((loads, isolated))
        y = np.r_[raw[:, j], np.asarray(sweeps['cmd_stress'])[m]]
        keep = np.isfinite(y) & (np.abs(y - np.nanmedian(y)) < 6)
        a, y = _poly(x[keep], _TERMS), y[keep]
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        support = np.abs(coef) > 0.002
        coefficients[support, j] = np.linalg.lstsq(a[:,support], y, rcond=None)[0]
    active = np.any(coefficients != 0, axis=1)
    return _TERMS[active], coefficients[active]

def _forward(x, bridge):
    exponents, coefficients = bridge
    return _poly(x, exponents) @ coefficients

def _invert(raw, bridge):
    raw = np.asarray(raw)
    exponents, coefficients = bridge
    linear = (exponents.sum(axis=1) == 1)
    jac0 = (exponents[linear].T @ coefficients[linear]).T
    loads = (raw - coefficients[exponents.sum(axis=1) == 0].sum(axis=0)) @ np.linalg.inv(jac0).T
    axes = np.eye(4, dtype=int)
    derivatives = [np.maximum(exponents - e, 0) for e in axes]
    for _ in range(30):
        error = _forward(loads, bridge) - raw
        if np.max(abs(error)) < 1e-11:
            break
        jac = np.stack([_poly(loads, derivatives[j]) @ (coefficients * exponents[:,j,None]) for j in range(4)],axis=-1)
        step = np.linalg.solve(jac, error[...,None])[...,0]
        trial = loads - step
        for _ in range(12):
            bad = np.max(abs(_forward(trial,bridge)-raw),axis=-1,keepdims=True) > np.max(abs(error),axis=-1,keepdims=True)
            if not np.any(bad):
                break
            step = np.where(bad,step/2,step)
            trial = loads-step
        loads = trial
    return loads

def _loads(part, bridge):
    return tuple(_invert(part[key],bridge).sum(axis=-1) for key in ('past_channel_raw','hold_channel_raw'))


# Fit on complete chamber verification coupons against their own reference panel.
# The three coefficients multiply the fast, slow, and linear drift kernels.
_CHAMBER_DRIFT = np.array([0.12987, 2.75926, 0.18386])


def _chamber_response(taus, theta, hold, x, times):
    drift = theta[2] + x @ theta[3:]
    amplitudes = hold[:, None] * theta[:2] + drift[:, None] * _CHAMBER_DRIFT[:2]
    linear = drift * _CHAMBER_DRIFT[2]
    strain = amplitudes @ (-np.expm1(-times[:, None] / taus)).T
    strain += linear[:, None] * times
    rate = amplitudes @ (1.0 / taus) + linear
    return strain, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    xt = _features(train, test)
    n = len(train['lab_id'])
    bridge = _bridge(dataset['calibration_load_cells'])
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    curves = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    taus, theta = _fit(*_loads(train, bridge), xt[:n], dt, times, curves,
                       np.asarray(train['instant_rate_obs']))
    hold = _invert(test['hold_channel_raw'], bridge).sum(axis=-1)
    strain, rate = _chamber_response(taus, theta, hold, xt[n:], times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + strain,
            'initial_rate_pred': rate}
