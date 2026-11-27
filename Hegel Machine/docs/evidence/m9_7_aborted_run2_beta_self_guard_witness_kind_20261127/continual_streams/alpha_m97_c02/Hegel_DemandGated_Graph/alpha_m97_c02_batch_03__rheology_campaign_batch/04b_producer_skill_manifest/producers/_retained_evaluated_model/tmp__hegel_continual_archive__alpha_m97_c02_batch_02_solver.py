"""Fit the two-branch linear-flow rheology model from calibrated observations."""

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data):
    c = data['calibration_load_cells']
    result = np.empty((4, 2))
    for lab in range(4):
        take = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][take], c['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            clean = np.abs(y - gain*x - np.median(y - gain*x)) <= 3.0
        if clean.sum() >= 2 and (clean.all() or clean.sum() >= 4):
            result[lab] = _line(x[clean], y[clean])
        else:
            result[lab] = _archive_calibration(lab, x[clean], y[clean])
    return result


def _archive_calibration(lab, x, y):
    import json
    try:
        with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as file:
            records = json.load(file)
    except FileNotFoundError:
        return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)

    def rows(node):
        if isinstance(node, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= node.keys():
                yield from zip(*(np.atleast_1d(node[key]) for key in ('lab_id', 'cmd_stress', 'true_stress')))
            else:
                for value in node.values():
                    yield from rows(value)
        elif isinstance(node, list):
            for value in node:
                yield from rows(value)

    pairs = np.array([(u, v) for c, u, v in rows(records) if int(c) == lab and np.isfinite(u) and np.isfinite(v)]).reshape(-1, 2)
    if len(pairs) >= 2:
        gain, bias = _line(*pairs.T)
        if len(x):
            bias = np.mean(y - gain*x)
        return gain, bias
    return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)


def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return split['past_stress_cmd']*gain[:, None] + bias[:, None], split['stress_hold_cmd']*gain + bias


def _design(logtau, past, hold, dt, time, features, derivatives=False):
    tau = np.exp(logtau)
    u = dt[:, None]/tau
    decay = np.exp(-u)
    update = -np.expm1(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for j in range(len(dt)):
        if derivatives:
            dz = dz*decay[j] + (z - past[:, j, None])*(u[j]*decay[j])
        z = z*decay[j] + past[:, j, None]*update[j]
    gap = hold[:, None] - z
    q = time[:, None]/tau
    accum = -np.expm1(-q)
    n, k = len(hold), len(time)
    curves = np.empty((n, k, 7))
    curves[:, :, :2] = gap[:, None, :]*accum
    curves[:, :, 2:] = features[:, None, :]*time[None, :, None]
    rates = np.column_stack((gap/tau, features))
    if not derivatives:
        return curves, rates
    dc = -dz[:, None, :]*accum - gap[:, None, :]*(q*np.exp(-q))
    dr = (-dz - gap)/tau
    return curves, rates, dc, dr


def _mad(x):
    return max(1.4826*np.median(np.abs(x - np.median(x))), 1e-7)


def _fit(past, hold, dt, time, features, strains, instant):
    y = np.r_[strains.ravel(), instant]
    n_curves = strains.size
    scale = np.ones_like(y)
    lower = np.r_[np.log(min(dt.min(), time[time > 0].min()) / 15), np.log(min(dt.min(), time[time > 0].min()) / 15), np.full(7, -np.inf)]
    upper = np.r_[np.log(40*max(time.max(), dt.sum())), np.log(40*max(time.max(), dt.sum())), np.full(7, np.inf)]

    def residual(p, jac=False):
        curves, rates, dc, dr = _design(p[:2], past, hold, dt, time, features, True)
        linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
        if jac:
            derivative = np.concatenate((dc.reshape(-1, 2), dr), axis=0)
            return np.column_stack((derivative*p[2:4], linear))/scale[:, None]
        return (linear @ p[2:] - y)/scale

    starts = [(0.35, 1.8), (0.8, 4.0), (0.18, 0.8), (1.5, 9.0)]
    start = np.log(starts[0])
    curves, rates = _design(start, past, hold, dt, time, features)
    linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
    pilot = np.r_[start, np.linalg.lstsq(linear, y, rcond=None)[0]]
    error = residual(pilot)

    def set_scale(err):
        scale[:n_curves] = 1.8*_mad(err[:n_curves])
        scale[n_curves:] = 1.8*_mad(err[n_curves:])

    set_scale(error)
    def optimize(p, loss):
        return least_squares(residual, p, jac=lambda p: residual(p, True),
                             bounds=(lower, upper), loss=loss, x_scale='jac', max_nfev=100)

    pilot = optimize(pilot, 'soft_l1').x
    set_scale(residual(pilot)*scale)
    best = optimize(pilot, 'cauchy')
    for pair in starts[1:]:
        start = np.log(pair)
        curves, rates = _design(start, past, hold, dt, time, features)
        linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
        weights = 1 / np.sqrt(1 + residual(best.x)**2) / scale
        p = np.r_[start, np.linalg.lstsq(linear*weights[:, None], y*weights, rcond=None)[0]]
        fit = optimize(p, 'cauchy')
        if fit.cost < best.cost:
            best = fit
    for _ in range(2):
        set_scale(residual(best.x)*scale)
        best = optimize(best.x, 'cauchy')
    p = best.x.copy()
    if p[0] > p[1]:
        p[:2] = p[1::-1]
        p[2:4] = p[3:1:-1]
    return p[:2], p[2:]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    cal = _calibration(dataset)
    aligned = _align(dataset)
    features = np.column_stack((np.ones(len(aligned)), aligned))
    n = len(train['lab_id'])
    past, hold = _physical(train, cal)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    strains = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, hw = _fit(past, hold, dt, time, features[:n], strains, train['instant_rate_obs'])
    past, hold = _physical(test, cal)
    curves, rates = _design(logtau, past, hold, dt, time, features[n:])
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': hw,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + curves @ hw,
            'initial_rate_pred': rates @ hw}
