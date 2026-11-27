import numpy as np
from scipy.optimize import least_squares


def basis(logtau, dt, t, past, hold, x, act, log_rows):
    tau = np.exp(logtau)
    h = dt[:, None] / tau
    a = np.exp(-h)
    b = -np.expm1(-h)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for s in range(len(dt)):
        u = past[:, s, None]
        dz = a[s] * (dz + h[s] * (z - u))
        z = a[s] * z + b[s] * u
    gap = hold[:, None] - z
    k = t[:, None] / tau
    phi = -np.expm1(-k)
    dphi = -k * np.exp(-k)
    flow = np.where(log_rows[:, None], np.log1p(t), t)
    drift = np.column_stack((np.ones(len(hold)), x))
    av = act * hold
    c = np.concatenate((gap[:, None] * phi, drift[:, None] * flow[:, :, None],
                        (av[:, None] * phi[:, 0])[:, :, None],
                        ((act * z[:, 0])[:, None] * flow)[:, :, None]), axis=2)
    r = np.column_stack((gap / tau, drift, av / tau[0], act * z[:, 0]))
    dc = -dz[:, None] * phi + gap[:, None] * dphi
    dr = (-dz - gap) / tau
    ec = np.stack((av[:, None] * dphi[:, 0], (act * dz[:, 0])[:, None] * flow), axis=-1)
    er = np.column_stack((-av / tau[0], act * dz[:, 0]))
    return c, r, dc, dr, ec, er


def fit(dt, t, past, hold, x, act, log_rows, yc, yr, robust=True):
    y = np.r_[yc.ravel(), yr]
    n = yc.size
    def design(logtau):
        c, r, dc, dr, ec, er = basis(logtau, dt, t, past, hold, x, act, log_rows)
        return (np.vstack((c.reshape(-1, 9), r)), np.vstack((dc.reshape(-1, 2), dr)),
                np.vstack((ec.reshape(-1, 2), er)))

    def run(p, sc, sr, loss):
        scale = np.r_[np.full(n, sc), np.full(len(yr), sr)]
        last = None
        def get(logtau):
            nonlocal last
            if last is None or not np.array_equal(logtau, last[0]):
                last = logtau.copy(), design(logtau)
            return last[1]
        def fun(p):
            a, _, _ = get(p[:2])
            return (np.einsum('ij,j->i', a, p[2:]) - y) / scale
        def jac(p):
            a, d, e = get(p[:2])
            j = d * p[2:4]
            j[:, 0] += e @ p[-2:]
            return np.column_stack((j, a)) / scale[:, None]
        low = np.log(min(dt.min(), t[t > 0].min()) / 20)
        high = np.log((dt.sum() + t.max()) * 20)
        return least_squares(fun, p, jac=jac, loss=loss, x_scale='jac', max_nfev=90,
                             bounds=(np.r_[[low]*2, [-np.inf]*9], np.r_[[high]*2, [np.inf]*9]))

    results = []
    for ts in ((.26, 1.4), (.55, 3.2), (.9, 5.5), (1.4, .26)):
        l = np.log(ts)
        a, _, _ = design(l)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        results.append(run(np.r_[l,w], .04, .05, 'cauchy' if robust else 'linear'))
    p = min(results, key=lambda s: s.cost).x
    if robust:
        a, _, _ = design(p[:2])
        e = np.einsum('ij,j->i', a, p[2:]) - y
        def scale(v):
            return max(1e-5, 2.5 * 1.4826 * np.median(np.abs(v - np.median(v))))
        p = run(p, scale(e[:n]), scale(e[n:]), 'cauchy').x
    return p


def _line(cmd, true):
    return np.linalg.lstsq(np.column_stack((cmd, np.ones(len(cmd)))), true, rcond=None)[0]


def _clean_calibration(cmd, true):
    if len(cmd) < 3:
        return np.ones(len(cmd), dtype=bool)
    i, j = np.triu_indices(len(cmd), 1)
    dx = cmd[j] - cmd[i]
    valid = dx != 0
    if not np.any(valid):
        return np.ones(len(cmd), dtype=bool)
    slopes = (true[j[valid]] - true[i[valid]]) / dx[valid]
    offsets = true[:, None] - cmd[:, None] * slopes
    deviations = np.abs(offsets - np.median(offsets, axis=0))
    return deviations[:, np.argmin(np.median(deviations, axis=0))] <= 3.0


def _archived_calibration(channel):
    import json
    from pathlib import Path

    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    for sweep in reversed(json.loads(path.read_text())):
        lab = np.asarray(sweep['lab_id'])
        cmd = np.asarray(sweep['cmd_stress'])[lab == channel]
        true = np.asarray(sweep['true_stress'])[lab == channel]
        clean = _clean_calibration(cmd, true)
        if np.unique(cmd[clean]).size >= 2:
            return _line(cmd[clean], true[clean])
    raise ValueError(f'No archived calibration for channel {channel}')


def _calibrate(cmd, true, channel):
    if len(cmd) in (3, 4) and np.unique(cmd).size == len(cmd):
        # In a short sweep, compare the fits left after omitting each row.
        # Three surviving rows identify the clean line; two cannot do so
        # without a prior. Prediction leverage accounts for calibration
        # rounding when the omitted command is far from the surviving pair.
        n = len(cmd)
        keep = np.broadcast_to(np.arange(n), (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)
        x, y = cmd[keep], true[keep]
        xm, ym = x.mean(axis=1), y.mean(axis=1)
        xc = x - xm[:, None]
        ss = np.sum(xc**2, axis=1)
        slope = np.sum(xc * (y - ym[:, None]), axis=1) / ss
        fit_error = np.linalg.norm(y - ym[:, None] - slope[:, None] * xc, axis=1)
        omitted_error = np.abs(true - ym - slope * (cmd - xm))
        leverage = np.sqrt(1 + 1 / x.shape[1] + (cmd - xm)**2 / ss)
        spike = omitted_error > 3.0 - 0.0003 * leverage
        if np.any(spike):
            if len(cmd) == 3:
                gain, bias = _archived_calibration(channel)
                best = np.argmax(np.abs(true - (gain * cmd + bias)))
            else:
                best = np.argmin(fit_error)
            if spike[best]:
                return _line(x[best], y[best])
        return _line(cmd, true)
    clean = _clean_calibration(cmd, true)
    if clean.all() and len(cmd) >= 2:
        return _line(cmd, true)
    if np.count_nonzero(clean) >= 4:
        return _line(cmd[clean], true[clean])
    gain, bias = _archived_calibration(channel)
    if not clean.all():
        archive_clean = np.abs(true - (gain * cmd + bias)) <= 3.0
        if archive_clean.sum() >= clean.sum():
            clean = archive_clean
    if np.unique(cmd[clean]).size >= 2:
        return _line(cmd[clean], true[clean])
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd']))
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]
    for channel in np.unique(lab):
        rows = cal['lab_id'] == channel
        gain, bias = _calibrate(cal['cmd_stress'][rows], cal['true_stress'][rows], channel)
        take = lab == channel
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - center) / scale
        for level in np.unique(stratum[take]):
            cell = take & (stratum == level)
            values = x[cell]
            x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    plateau = dataset['shear_banding_index'] <= 0.50
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)**2 @ dt
    act = np.tanh(damage) if plateau else np.zeros_like(damage)
    # Low-banding campaigns have logarithmic baseline flow. Damage gates the
    # sample-specific crystallization terms, not the baseline time dependence.
    return dt, t, past, hold, x, act, np.full(len(lab), plateau, dtype=bool)


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    dt, t, *samples = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    p = fit(dt, t, *(a[:n] for a in samples), yc, train['instant_rate_obs'])
    c, r, *_ = basis(p[:2], dt, t, *(a[n:] for a in samples))
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:9],
                regime_mode_weights=p[9:], predicted_strain=test['strain_0'][:, None] + c @ p[2:],
                initial_rate_pred=r @ p[2:])
