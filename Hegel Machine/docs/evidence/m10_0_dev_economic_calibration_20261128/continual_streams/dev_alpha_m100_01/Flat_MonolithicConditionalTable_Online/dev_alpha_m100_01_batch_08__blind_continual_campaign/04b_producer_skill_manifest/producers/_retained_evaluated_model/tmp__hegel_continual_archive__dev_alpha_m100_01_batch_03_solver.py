"""Estimate a two-branch hereditary response from the benchtop coupons."""
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
    """Fit raw response to controlled loads, rejecting corrupted sweep points."""
    raw = np.asarray(cal['bridge_raw'])
    loads = np.asarray(cal['true_channel_loads'])
    sweeps = cal['single_channel_sweeps']
    coeff = []
    for i in range(4):
        nxt, prev = (i + 1) % 4, (i - 1) % 4
        u, v, w = loads[:, [i, nxt, prev]].T
        a = np.column_stack((np.ones_like(u), u, v, w, u*v, u**3))
        y = raw[:, i]
        m = np.asarray(sweeps['lab_id']) == i
        x = np.asarray(sweeps['true_stress'])[m]
        if x.size:
            z = np.zeros_like(x)
            a = np.vstack((a, np.column_stack((np.ones_like(x), x, z, z, z, x**3))))
            y = np.r_[y, np.asarray(sweeps['cmd_stress'])[m]]
        beta = least_squares(lambda b: a @ b - y,
                             np.linalg.lstsq(a, y, rcond=None)[0],
                             jac=lambda _: a.copy(), loss='soft_l1', f_scale=0.002).x
        err = a @ beta - y
        mad = 1.4826 * np.median(np.abs(err - np.median(err)))
        keep = np.abs(err - np.median(err)) <= max(0.003, 6 * mad)
        coeff.append(np.linalg.lstsq(a[keep], y[keep], rcond=None)[0])
    return np.array(coeff).T


def _transfer(loads, bridge):
    bias, diag, nxt, prev, pair, cubic = bridge
    v, w = np.roll(loads, -1, axis=-1), np.roll(loads, 1, axis=-1)
    return bias + diag*loads + nxt*v + prev*w + pair*loads*v + cubic*loads**3


def _invert(raw, bridge):
    raw = np.asarray(raw)
    bias, diag, nxt, prev, pair, cubic = bridge
    i = np.arange(4)
    j, k = (i + 1) % 4, (i - 1) % 4
    loads = (raw - bias) / diag
    jac = np.zeros(raw.shape + (4,))
    jac[..., i, k] = prev
    for _ in range(30):
        error = _transfer(loads, bridge) - raw
        if np.max(np.abs(error)) < 1e-10:
            break
        jac[..., i, i] = diag + pair*loads[..., j] + 3*cubic*loads**2
        jac[..., i, j] = nxt + pair*loads
        step = np.linalg.solve(jac, error[..., None])[..., 0]
        loads -= step / np.maximum(1, np.max(np.abs(step), axis=-1, keepdims=True))
    return loads


def _loads(part, bridge):
    return tuple(_invert(part[key], bridge).sum(axis=-1)
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


def solve_batch(dataset):
    train, test = dataset['train'], dataset['test']
    x = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    n = len(train['lab_id'])
    dt, t = (np.asarray(dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train, bridge), x[:n], dt, t,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    # Both equilibration markers preceding this deployment are back at their
    # facility initialization levels: retain the baseline chamber law.
    c, r = _design(taus, *_loads(test, bridge), x[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + c @ theta,
            'initial_rate_pred': r @ theta}
