import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = xc @ (y - y.mean()) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _spikes(x, y):
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not len(i):
        return np.zeros(len(x), dtype=bool)
    gains = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[:, None] - x[:, None] * gains
    errors = np.abs(offsets - np.median(offsets, axis=0))
    best = np.argmin(np.median(errors, axis=0))
    return errors[:, best] > 3.0


def _prior_line(channel):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        row = np.asarray(sweep['lab_id']) == channel
        x = np.asarray(sweep['cmd_stress'])[row]
        y = np.asarray(sweep['true_stress'])[row]
        keep = ~_spikes(x, y)
        if np.unique(x[keep]).size >= 2:
            return _line(x[keep], y[keep])
    raise ValueError(f'No usable calibration sweep for channel {channel}')


def _calibrate(x, y, channel):
    if len(x) >= 4:
        keep = ~_spikes(x, y)
        if keep.sum() >= 4 or keep.all():
            return _line(x[keep], y[keep])
    else:
        keep = np.ones(len(x), dtype=bool)
        if len(x) == 3:
            # Two surviving points cannot identify which of three rows spiked.
            # Look for a >3 discrepancy before consulting the prior sweep.
            errors = [abs(y[k] - np.polyval(_line(np.delete(x, k), np.delete(y, k)), x[k]))
                      for k in range(3) if np.unique(np.delete(x, k)).size == 2]
            slope, bias = _line(x, y)
            if not errors or max(errors) <= 3 or np.max(np.abs(y - slope*x - bias)) < .01:
                return slope, bias
        elif len(x) == 2:
            return _line(x, y)
    gain, bias = _prior_line(channel)
    if len(x) >= 3 and keep.all():
        keep = np.abs(y - (gain*x + bias)) <= 3.0
    if np.unique(x[keep]).size >= 2:
        clean_x, clean_y = x[keep], y[keep]
        fitted_gain, fitted_bias = _line(clean_x, clean_y)
        # With fewer than four clean readings after EMI rejection, borrow
        # the archived slope only if supported by the in-batch contrasts.
        # Keep the stress level anchored to the current measurements.
        x_centered = clean_x - clean_x.mean()
        if keep.sum() < 4:
            contrasts = clean_y - clean_y.mean() - gain * x_centered
            if np.ptp(contrasts) <= 0.001:
                return gain, clean_y.mean() - gain * clean_x.mean()
        return fitted_gain, fitted_bias
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd'])).astype(float)
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']].astype(float)
    for c in np.unique(lab):
        rows = np.asarray(cal['lab_id']) == c
        gain, bias = _calibrate(np.asarray(cal['cmd_stress'])[rows], np.asarray(cal['true_stress'])[rows], c)
        take = lab == c
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        x[take] = (x[take] - np.nanmedian(np.nanmean(panel, axis=0))) / np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
    cells = []
    for c in np.unique(lab):
        for stratum_id in np.unique(stratum[lab == c]):
            take = (lab == c) & (stratum == stratum_id)
            valid = np.isfinite(x[take]).sum(axis=0)
            mean = np.divide(np.nansum(x[take], axis=0), valid, out=np.full(x.shape[1], np.nan), where=valid != 0)
            cells.append((c, take, mean))
    for c, take, mean in cells:
        if np.isnan(mean).any():
            with np.errstate(invalid='ignore', divide='ignore'):
                local = np.array([v for lab_id, _, v in cells if lab_id == c])
                count = np.isfinite(local).sum(axis=0)
                local = np.divide(np.nansum(local, axis=0), count, out=np.zeros(x.shape[1]), where=count != 0)
            mean = np.where(np.isnan(mean), local, mean)
        x[take] = np.where(np.isnan(x[take]), mean, x[take])
    return np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times']), past, hold, x


def _basis(tau, dt, t, past, hold, x):
    h = dt[:, None] / tau
    decay = np.exp(-h)
    drive = -np.expm1(-h)
    memory = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        memory = decay[s] * memory + drive[s] * past[:, s, None]
    gap = hold[:, None] - memory
    contrast = gap[:, 0] - gap[:, 1]
    drift_slip = np.column_stack((np.ones(len(hold)), x, np.sign(hold) * np.sqrt(np.abs(hold))))
    relaxation = -np.expm1(-t[:, None] / tau)
    flow = t / (1 + 0.85 * t)
    curves = np.concatenate((gap[:, None, :] * relaxation,
                             drift_slip[:, None, :] * flow[None, :, None],
                             contrast[:, None, None] * relaxation[None, :, :1]), axis=2)
    rates = np.column_stack((gap / tau, drift_slip, contrast / tau[0]))
    return curves, rates


def _fit(dt, t, past, hold, x, yc, yr):
    y = np.r_[yc.ravel(), yr]
    low = min(dt.min(), t[t > 0].min()) / 20
    high = (dt.sum() + t.max()) * 20
    # Search ordered branches directly: the ninth basis term is tied to the fast branch.
    def decode(v):
        fast, separation = np.exp(v)
        return np.array([fast, fast + separation])

    def design(v):
        c, r = _basis(decode(v), dt, t, past, hold, x)
        return np.concatenate((c.reshape(-1, 9), r))

    def profile(v):
        a = design(v)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, w) - y, w

    best = None
    for fast, separation in ((0.5, 3), (0.2, 1), (1.0, 8)):
        v0 = np.log(np.clip([fast, separation], low, high))
        opt = least_squares(lambda v: profile(v)[0], v0,
                            bounds=(np.log(low), np.log(high)), max_nfev=100)
        if best is None or opt.cost < best.cost:
            best = opt
    initial = np.r_[best.x, profile(best.x)[1]]
    n_curve = yc.size
    scale = np.ones_like(y)

    def residual(p):
        return np.einsum('ij,j->i', design(p[:2]), p[2:]) - y

    def mad(e):
        return max(1e-8, 1.4826 * np.median(np.abs(e - np.median(e))))

    bounds = (np.r_[[np.log(low)] * 2, [-np.inf] * 9],
              np.r_[[np.log(high)] * 2, [np.inf] * 9])
    p = initial
    # Fit separate noise scales for the curve and instantaneous-rate instruments.
    # The Cauchy loss limits the influence of transient curve spikes.
    for _ in range(3):
        e = residual(p)
        scale[:n_curve] = mad(e[:n_curve])
        scale[n_curve:] = mad(e[n_curve:])
        p = least_squares(lambda q: residual(q) / scale, p, bounds=bounds,
                          loss='cauchy', f_scale=2, max_nfev=120).x
    return decode(p[:2]), p[2:]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    dt, t, past, hold, x = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['strain_0'])
    yc = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    yr = np.asarray(train['instant_rate_obs'])
    tau, w = _fit(dt, t, past[:n], hold[:n], x[:n], yc, yr)
    curves, rates = _basis(tau, dt, t, past[n:], hold[n:], x[n:])
    return {'estimated_taus': tau, 'hereditary_weights': w[:7], 'regime_mode_weights': w[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + curves @ w,
            'initial_rate_pred': rates @ w}
