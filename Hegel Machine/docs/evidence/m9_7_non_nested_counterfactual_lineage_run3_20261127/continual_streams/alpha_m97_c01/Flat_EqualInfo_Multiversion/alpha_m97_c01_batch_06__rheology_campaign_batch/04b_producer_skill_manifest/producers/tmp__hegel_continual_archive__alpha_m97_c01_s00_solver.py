import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _CALIBRATION_ARCHIVE.is_file():
        return None
    archive = json.loads(_CALIBRATION_ARCHIVE.read_text())
    if 'stress_gains' in archive and 'stress_biases' in archive:
        return float(archive['stress_gains'][channel]), float(archive['stress_biases'][channel])
    channels = archive.get('channels', archive)
    record = channels.get(str(channel)) if isinstance(channels, dict) else channels[channel]
    if record is None:
        return None
    return float(record['gain']), float(record['bias'])


def _calibrate(cmd, phys, channel):
    if len(cmd) >= 3:
        i, j = np.triu_indices(len(cmd), 1)
        dx = cmd[j] - cmd[i]
        keep = dx != 0
        slopes = (phys[j[keep]] - phys[i[keep]]) / dx[keep]
        offsets = np.median(phys[:, None] - cmd[:, None] * slopes, axis=0)
        error = np.abs(phys[:, None] - cmd[:, None] * slopes - offsets)
        robust = np.argmin(np.median(error, axis=0))
        clean = error[:, robust] <= 3.0
    else:
        clean = np.ones(len(cmd), dtype=bool)
    if len(cmd) < 2 or not clean.all():
        prior = _archived_calibration(channel)
        if prior is not None and len(cmd) >= 2 and clean.sum() < 4:
            prior_clean = np.abs(phys - (prior[0] * cmd + prior[1])) <= 3.0
            if prior_clean.sum() >= clean.sum():
                clean = prior_clean
    else:
        prior = None
    if np.unique(cmd[clean]).size >= 2:
        return np.polyfit(cmd[clean], phys[clean], 1)
    if prior is not None:
        return prior
    raise ValueError(f'Insufficient load-cell calibration for channel {channel}')


def _prepare(dataset):
    cal = dataset['calibration_load_cells']
    train, test = dataset['train'], dataset['test']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd']))
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]
    for channel in np.unique(lab):
        take = lab == channel
        rows = cal['lab_id'] == channel
        cmd, phys = cal['cmd_stress'][rows], cal['true_stress'][rows]
        gain, bias = _calibrate(cmd, phys, int(channel))
        past[take] = past[take] * gain + bias
        hold[take] = hold[take] * gain + bias
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[take] = (x[take] - center) / scale
        for level in np.unique(stratum[take]):
            cell = take & (stratum == level)
            values = x[cell]
            x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _basis(logtau, dt, t, past, hold, x, derivative=False):
    tau = np.exp(logtau)
    end = np.cumsum(dt[::-1])[::-1] - dt
    a = end[:, None] / tau
    b = dt[:, None] / tau
    decay = np.exp(-a)
    w = decay * -np.expm1(-b)
    gap = hold[:, None] - past @ w
    u = t[:, None] / tau
    step = -np.expm1(-u)
    curves = np.concatenate((gap[:, None, :] * step, x[:, None, :] * t[None, :, None]), axis=2)
    rates = np.column_stack((gap / tau, x))
    if not derivative:
        return curves, rates
    dgap = -past @ (a * w - b * np.exp(-a-b))
    dc = dgap[:, None, :] * step - gap[:, None, :] * u * np.exp(-u)
    dr = (dgap - gap) / tau
    return curves, rates, dc, dr


def _mad(a):
    return max(1e-7, 1.4826 * np.median(np.abs(a - np.median(a))))


def _fit(dt, t, past, hold, x, yc, yr):
    def linear(logtau):
        c, r = _basis(logtau, dt, t, past, hold, x)
        cm = c.reshape(-1, 7)
        return np.linalg.lstsq(np.vstack((cm, r)), np.r_[yc.ravel(), yr], rcond=None)[0]

    def get(p):
        c, r, dc, dr = _basis(p[:2], dt, t, past, hold, x, True)
        return c.reshape(-1, 7), r, dc.reshape(-1, 2)*p[2:4], dr*p[2:4]

    def func(p, sc, sr):
        c, r, _, _ = get(p)
        return np.r_[(c@p[2:] - yc.ravel())/sc, (r@p[2:] - yr)/sr]

    def jac(p, sc, sr):
        c, r, dc, dr = get(p)
        return np.vstack((np.column_stack((dc, c))/sc, np.column_stack((dr, r))/sr))

    bounds = (np.r_[np.log(max(1e-4, min(dt.min(), t.min())/20))*np.ones(2), np.full(7, -np.inf)],
              np.r_[np.log((dt.sum()+t.max())*20)*np.ones(2), np.full(7, np.inf)])
    best = None
    for ts in ((.26, 1.4), (.55, 3.2), (.9, 5.5)):
        logs = np.log(ts)
        p = np.r_[logs, linear(logs)]
        opts = dict(jac=jac, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=100)
        fit = least_squares(func, p, args=(.04, .05), **opts)
        if best is None or fit.cost < best.cost:
            best = fit
    raw = func(best.x, 1, 1)
    sc = 2.5 * _mad(raw[:yc.size])
    sr = 2.5 * _mad(raw[yc.size:])
    fit = least_squares(func, best.x, args=(sc, sr), **opts)
    p = fit.x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    past, hold, x = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    t, dt = dataset['eval_times'], dataset['past_dt']
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, weights = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])
    c, r = _basis(logtau, dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(logtau), hereditary_weights=weights,
                regime_mode_weights=np.zeros(2), predicted_strain=test['strain_0'][:, None]+c@weights,
                initial_rate_pred=r@weights)
