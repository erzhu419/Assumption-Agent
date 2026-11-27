"""Calibrate the four rheometers and fit the jammed-slip constitutive law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _prior_calibration(channel):
    # The two immediately preceding sweeps share this campaign's calibration.
    sweeps = json.loads(_ARCHIVE.read_text())[-2:]
    cmd, true = [], []
    for sweep in sweeps:
        rows = np.asarray(sweep['lab_id']) == channel
        cmd.extend(np.asarray(sweep['cmd_stress'])[rows])
        true.extend(np.asarray(sweep['true_stress'])[rows])
    return np.polyfit(cmd, true, 1)


def _calibrate(cmd, true, channel):
    if np.unique(cmd).size < 2:
        return _prior_calibration(channel)
    direct = np.polyfit(cmd, true, 1)
    if len(cmd) == 2:
        # Two points give zero affine residual even if one is an EMI spike.
        # Compare against the compatible prefix sweep to detect the spike,
        # but retain the in-batch fit when neither point exceeds the threshold.
        prior = _prior_calibration(channel)
        if np.any(np.abs(true - np.polyval(prior, cmd)) > 3.0):
            return prior
        return direct
    # Clean public sweeps are affine within 0.00022 stress units. A loose
    # consensus tolerance separates clean points from plausible EMI without
    # assuming that the least-squares line is unbiased in a sparse sweep.
    consensus_tol = 0.01
    # In a sparse sweep, two nearby commands can let an EMI spike at a
    # third command hide behind a nearly exact but implausible affine fit.
    # The compatible campaign sweeps have gains 0.77–1.34 and |bias| < 0.18.
    # Use a generous plausibility range only to flag sparse fits, never to
    # clip or replace a calibration supported by four clean observations.
    plausible_sparse_fit = 0.3 < direct[0] < 2.0 and abs(direct[1]) < 0.6
    if (len(cmd) >= 4 or plausible_sparse_fit) and np.max(np.abs(true - np.polyval(direct, cmd))) <= consensus_tol:
        return direct
    i, j = np.triu_indices(len(cmd), 1)
    distinct = cmd[i] != cmd[j]
    i, j = i[distinct], j[distinct]
    slopes = (true[j] - true[i]) / (cmd[j] - cmd[i])
    offsets = true[i] - cmd[i] * slopes
    errors = np.abs(true[:, None] - cmd[:, None] * slopes - offsets)
    support = np.sum(errors <= consensus_tol, axis=0)
    best = np.argmax(support)
    clean = errors[:, best] <= consensus_tol
    if clean.sum() >= 4:
        return np.polyfit(cmd[clean], true[clean], 1)
    # A noisy three-point sweep cannot distinguish competing two-point lines.
    prior = _prior_calibration(channel)
    clean = np.abs(true - np.polyval(prior, cmd)) <= 3.0
    if clean.all():
        return direct
    if np.unique(cmd[clean]).size >= 2:
        return np.polyfit(cmd[clean], true[clean], 1)
    return prior


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd']))
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]
    for channel in np.unique(lab):
        rows = cal['lab_id'] == channel
        gain, bias = _calibrate(cal['cmd_stress'][rows], cal['true_stress'][rows], channel)
        m = lab == channel
        past[m] = gain * past[m] + bias
        hold[m] = gain * hold[m] + bias
        panel = refs[m]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (x[m] - center) / scale
        for level in np.unique(stratum[m]):
            cell = m & (stratum == level)
            values = x[cell]
            if np.isnan(values).any():
                x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _basis(logtau, dt, t, past, hold, x, jac=False):
    tau = np.exp(logtau)
    after = (np.cumsum(dt[::-1])[::-1] - dt)[:, None] / tau
    step_dt = dt[:, None] / tau
    memory_weights = np.exp(-after) * -np.expm1(-step_dt)
    gap = hold[:, None] - past @ memory_weights
    u = t[:, None] / tau
    step = -np.expm1(-u)
    flow = t / (1.0 + 0.85 * t)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    diff = gap[:, 0] - gap[:, 1]
    c = np.concatenate((gap[:, None, :] * step,
                        x[:, None, :] * flow[None, :, None],
                        (slip[:, None] * flow)[:, :, None],
                        (diff[:, None] * step[:, 0])[:, :, None]), axis=2)
    r = np.column_stack((gap / tau, x, slip, diff / tau[0]))
    if not jac:
        return c, r
    dgap = -past @ (after * memory_weights - step_dt * np.exp(-after - step_dt))
    dstep = -u * np.exp(-u)
    dc = dgap[:, None, :] * step + gap[:, None, :] * dstep
    dr = (dgap - gap) / tau
    dcross = np.stack((dgap[:, 0, None] * step[:, 0] + diff[:, None] * dstep[:, 0],
                       -dgap[:, 1, None] * step[:, 0]), axis=2)
    drcross = np.column_stack((dgap[:, 0] - diff, -dgap[:, 1])) / tau[0]
    return c, r, dc, dr, dcross, drcross


def _fit(dt, t, past, hold, x, yc, yr):
    args = (dt, t, past, hold, x)
    yc = yc.ravel()
    target = np.r_[yc, yr]

    def linear(logs):
        c, r = _basis(logs, *args)
        a = np.vstack((c.reshape(-1, 9), r))
        return np.linalg.lstsq(a, target, rcond=None)[0]

    def residual(p, sc, sr):
        c, r = _basis(p[:2], *args)
        return np.r_[((c @ p[2:]).ravel() - yc) / sc, (r @ p[2:] - yr) / sr]

    def derivative(p, sc, sr):
        c, r, dc, dr, dcross, drcross = _basis(p[:2], *args, jac=True)
        jc = dc * p[2:4] + dcross * p[-1]
        jr = dr * p[2:4] + drcross * p[-1]
        return np.vstack((np.column_stack((jc.reshape(-1, 2), c.reshape(-1, 9))) / sc,
                          np.column_stack((jr, r)) / sr))

    low = np.log(min(dt.min(), t[t > 0].min()) / 20)
    high = np.log((dt.sum() + t.max()) * 20)
    bounds = (np.r_[[low, low], np.full(9, -np.inf)],
              np.r_[[high, high], np.full(9, np.inf)])
    opts = dict(jac=derivative, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=150)
    fits = []
    for pair in ((.26, 1.4), (.55, 3.2), (.9, 5.5), (1.5, .4)):
        logs = np.clip(np.log(pair), low + 1e-6, high - 1e-6)
        fits.append(least_squares(residual, np.r_[logs, linear(logs)], args=(.04, .05), **opts))
    best = min(fits, key=lambda fit: fit.cost)
    raw = residual(best.x, 1.0, 1.0)

    def scale(v):
        return max(1e-7, 2.5 * 1.4826 * np.median(np.abs(v - np.median(v))))

    refined = least_squares(residual, best.x, args=(scale(raw[:yc.size]), scale(raw[yc.size:])), **opts)
    return refined.x[:2], refined.x[2:]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    past, hold, x = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, w = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])
    c, r = _basis(logs, dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(logs), hereditary_weights=w[:7], regime_mode_weights=w[7:],
                predicted_strain=test['strain_0'][:, None] + c @ w,
                initial_rate_pred=r @ w)
