"""Calibrated two-branch hereditary model for the linear-flow campaign."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _archive_pairs():
    try:
        with open(_ARCHIVE) as handle:
            history = json.load(handle)
    except FileNotFoundError:
        return {}
    if isinstance(history, dict):
        history = history.get('calibration_load_cells', history.get('channels', history))
        if 'lab_id' in history and isinstance(history['lab_id'], list):
            keys = tuple(history)
            history = [dict(zip(keys, values)) for values in zip(*(history[k] for k in keys))]
        else:
            history = [dict(v, lab_id=v.get('lab_id', k)) for k, v in history.items() if isinstance(v, dict)]
    result = {}
    for row in history:
        if 'gain' in row and 'bias' in row:
            result[int(row['lab_id'])] = (float(row['gain']), float(row['bias']))
    return result


def _calibrate(cal):
    gain, bias = np.ones(4), np.zeros(4)
    archive = None
    for lab in range(4):
        use = cal['lab_id'] == lab
        x, y = cal['cmd_stress'][use], cal['true_stress'][use]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            different = np.abs(dx) > 1e-9
            if different.any():
                a = np.median((y[j] - y[i])[different] / dx[different])
                b = np.median(y - a*x)
                clean = np.abs(y - a*x - b) <= 3.0
        needs_archive = len(x) < 2 or not clean.all()
        if needs_archive and archive is None:
            archive = _archive_pairs()
        prior = archive.get(lab) if needs_archive else None
        if clean.sum() >= 4 or (clean.sum() >= 2 and prior is None):
            gain[lab], bias[lab] = np.linalg.lstsq(np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
        elif prior is not None:
            gain[lab], bias[lab] = prior
        elif clean.sum() == 1:
            bias[lab] = (y - x)[clean][0]
    return gain, bias


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _design(split, x, gain, bias, dt, t):
    g, b = gain[split['lab_id']], bias[split['lab_id']]
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    # Each past segment's surviving contribution to the terminal memory.
    age = np.cumsum(dt[::-1])[::-1] - dt
    flow = np.column_stack((np.ones(len(x)), x))
    curve = np.empty((len(x), len(t), 7))
    rate = np.empty((len(x), 7))
    curve[:, :, 2:] = flow[:, None, :]*t[None, :, None]
    rate[:, 2:] = flow

    def evaluate(logtau, deriv=False):
        tau = np.exp(logtau)
        h, a = dt[:, None]/tau, age[:, None]/tau
        survival = np.exp(-a)
        kernel = survival*(-np.expm1(-h))
        gap = hold[:, None] - hist @ kernel
        u = t[:, None]/tau
        envelope = -np.expm1(-u)
        curve[:, :, :2] = gap[:, None, :]*envelope
        rate[:, :2] = gap/tau
        if deriv:
            dgap = -hist @ (a*kernel - survival*h*np.exp(-h))
            dc = dgap[:, None, :]*envelope - gap[:, None, :]*u*np.exp(-u)
            dr = (dgap - gap)/tau
            return curve, rate, dc, dr
        return curve, rate

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    n = cm.sum()
    lo = np.log(min(dt.min(), t[t > 0].min()) / 20)
    hi = np.log(20*max(t.max(), dt.sum()))

    def matrices(logtau):
        c, r = evaluate(logtau)
        return np.concatenate((c[cm], r[rm]))

    def mad(r):
        return max(1e-7, 1.4826*np.median(np.abs(r - np.median(r)))) if len(r) else 1.0

    # Profile the linear coefficients at each deterministic starting pair.
    fast = min(np.median(dt), t[t > 0].min())
    slow = t.max()/2
    starts = [(fast, slow), (fast/3, slow), (2*fast, 2*slow),
              (fast, slow/3), (slow/2, slow*2)]

    def run(initial, scales):
        def residual(p):
            return (matrices(p[:2]) @ p[2:] - y)/scales

        def jacobian(p):
            c, r, dc, dr = evaluate(p[:2], True)
            M = np.concatenate((c[cm], r[rm]))
            D = np.concatenate((dc[cm], dr[rm]))*p[None, 2:4]
            return np.column_stack((D, M))/scales[:, None]

        return least_squares(residual, initial, jac=jacobian, loss='cauchy',
                             bounds=([lo]*2+[-np.inf]*7, [hi]*2+[np.inf]*7),
                             x_scale='jac', max_nfev=100)

    weights = np.r_[np.ones(n), np.full(len(y)-n, .5)]
    initial = []
    for start in starts:
        logs = np.clip(np.log(start), lo + 1e-6, hi - 1e-6)
        M = matrices(logs)
        w = weights.copy()
        for _ in range(7):
            beta = np.linalg.lstsq(M*w[:, None], y*w, rcond=None)[0]
            residual = M @ beta - y
            scale = max(.005, 2*mad(residual[:n]))
            w = weights / np.sqrt(1 + (residual*weights/scale)**2)
        initial.append(np.r_[logs, beta])
    coarse = np.r_[np.full(n, .06), np.full(len(y)-n, .12)]
    best = min((run(p, coarse) for p in initial), key=lambda opt: opt.cost)
    for _ in range(2):
        residual = matrices(best.x[:2]) @ best.x[2:] - y
        scales = np.r_[np.full(n, 1.6*mad(residual[:n])), np.full(len(y)-n, 1.6*mad(residual[n:]))]
        best = run(best.x, scales)
    order = np.argsort(best.x[:2])
    return best.x[:2][order], np.r_[best.x[2:4][order], best.x[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    build = _design(train, xtr, gain, bias, dt, t)
    logtau, beta = _fit(build, train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'], dt, t)
    curve, rate = _design(test, xte, gain, bias, dt, t)(logtau)
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': beta,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
