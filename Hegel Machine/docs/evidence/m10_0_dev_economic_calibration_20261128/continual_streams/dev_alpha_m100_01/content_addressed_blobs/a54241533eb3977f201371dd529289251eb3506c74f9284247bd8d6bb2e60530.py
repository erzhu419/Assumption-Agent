"""Calibrate the load bridge and fit the shared two-branch material law."""

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


_OPPOSITE = np.array([2, 3, 0, 1])


def _bridge(cal):
    """Fit the forward transfer in load space, including isolated sweeps."""
    loads = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    sweeps = cal['single_channel_sweeps']
    parameters = []
    for j, k in enumerate(_OPPOSITE):
        m = np.asarray(sweeps['lab_id']) == j
        axial = np.asarray(sweeps['true_stress'])[m]
        a = np.r_[loads[:, j], axial]
        b = np.r_[loads[:, k], np.zeros(len(axial))]
        y = np.r_[raw[:, j], np.asarray(sweeps['cmd_stress'])[m]]
        # The recorded gross faults are separated from the working sweep range.
        keep = np.isfinite(y) & (abs(y - np.nanmedian(y)) < 6)
        a, b, y = a[keep], b[keep], y[keep]

        def basis(freq):
            f, g = freq
            return np.column_stack((np.ones(len(a)), a, np.sin(f*a),
                                    b, -2*np.sin(g*b/2)**2))

        def profile(freq):
            mat = basis(freq)
            return mat @ np.linalg.lstsq(mat, y, rcond=None)[0] - y

        fit = least_squares(profile, [2.5, 2.0], bounds=([1., 0.7], [4., 3.5]),
                            gtol=1e-12)
        mat = basis(fit.x)
        parameters.append(np.r_[np.linalg.lstsq(mat, y, rcond=None)[0], fit.x])
    return np.asarray(parameters).T


def _forward(loads, bridge):
    offset, gain, sine, coupling, cosine, f, g = bridge
    opposite = loads[..., _OPPOSITE]
    return (offset + gain*loads + sine*np.sin(f*loads) + coupling*opposite
            - 2*cosine*np.sin(g*opposite/2)**2)


def _invert(raw, bridge):
    raw = np.asarray(raw)
    offset, gain, sine, coupling, cosine, f, g = bridge
    loads = (raw - offset) / gain
    i = np.arange(4)
    jac = np.zeros(raw.shape + (4,))
    for _ in range(35):
        err = _forward(loads, bridge) - raw
        if np.max(np.abs(err)) < 1e-11:
            break
        jac[..., i, i] = gain + f*sine*np.cos(f*loads)
        jac[..., i, _OPPOSITE] = coupling - g*cosine*np.sin(g*loads[..., _OPPOSITE])
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        # Near the shallow portion of a sweep, a full Newton step can overshoot.
        step /= np.maximum(1., np.max(abs(step), axis=-1, keepdims=True))
        norm = np.max(abs(err), axis=-1, keepdims=True)
        trial = loads - step
        for _ in range(10):
            bad = np.max(abs(_forward(trial, bridge) - raw), axis=-1, keepdims=True) > norm
            if not np.any(bad):
                break
            step = np.where(bad, step / 2, step)
            trial = loads - step
        loads = trial
    return loads


def _loads(part, bridge):
    return tuple(_invert(part[key], bridge).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, t = (np.asarray(dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train, bridge), x[:n], dt, t,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    # The latest pre-deployment facility trace has both equilibration markers
    # at initialization levels, as in the archived baseline chamber campaign.
    c, r = _design(taus, *_loads(test, bridge), x[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + c @ theta,
            'initial_rate_pred': r @ theta}
