"""Calibrated two-branch hereditary model for amorphous melt flow."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab = np.asarray(rows['lab_id'])
    cmd = np.asarray(rows['cmd_stress'], float)
    true = np.asarray(rows['true_stress'], float)
    coeff = np.empty((4, 2))
    archive_path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    for channel in range(4):
        m = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[m], true[m]
        i, j = np.triu_indices(x.size, 1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        clean = np.ones(x.size, dtype=bool)
        emi = False
        if len(i):
            slopes = (y[i] - y[j]) / (x[i] - x[j])
            lines = y[i, None] + slopes[:, None] * (x - x[i, None])
            err = np.abs(lines - y)
            # Estimate the affine relation from the tight majority first. A
            # count at the EMI cutoff alone favors lines tilted toward a spike.
            typical = np.median(err, axis=1)
            best = np.argmin(typical)
            core = err[best] <= 8 * typical[best] + 1e-12
            design = np.column_stack((x, np.ones(x.size)))
            provisional = np.linalg.lstsq(design[core], y[core], rcond=None)[0]
            residual = np.abs(design @ provisional - y)
            # Allow for calibration noise in deciding which near-cutoff rows
            # to exclude, without relaxing the trigger for archive lookup.
            margin = min(0.01, 8 * np.median(residual[core]))
            clean = residual <= 3.0 - margin
            emi = np.any(residual > 3.0)
        if (len(x) < 2 or emi) and clean.sum() < 4 and archive_path.is_file():
            with archive_path.open() as f:
                archive = json.load(f)
            historic = archive.get('calibration_load_cells', archive) if isinstance(archive, dict) else archive
            if isinstance(historic, dict) and 'lab_id' in historic:
                historic = [dict(zip(('lab_id', 'cmd_stress', 'true_stress'), r)) for r in zip(historic['lab_id'], historic['cmd_stress'], historic['true_stress'])]
            if isinstance(historic, list) and clean.sum() < 4:
                points = [(r['cmd_stress'], r['true_stress']) for r in historic
                          if {'lab_id', 'cmd_stress', 'true_stress'} <= r.keys() and int(r['lab_id']) == channel]
                if points:
                    xx, yy = np.asarray(points, float).T
                    x, y = np.r_[x[clean], xx], np.r_[y[clean], yy]
                    clean = np.ones(x.size, dtype=bool)
        x, y = x[clean], y[clean]
        if x.size >= 2 and np.ptp(x) > 0:
            coeff[channel] = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
        else:
            coeff[channel] = 1.0, np.median(y - x) if x.size else 0.0
    return coeff


def _inputs(dataset):
    splits = (dataset['train'], dataset['test'])

    def stack(key):
        return np.concatenate([s[key] for s in splits], axis=0)

    lab, strata = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel
        center = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        aligned[m] = (raw[m] - center) / scale
    cells = np.unique(np.column_stack((lab, strata)), axis=0)
    means = []
    for c, s in cells:
        v = aligned[(lab == c) & (strata == s)]
        count = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
    fill = np.nanmean(means, axis=0)
    aligned = np.where(np.isfinite(aligned), aligned, fill)
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dt, times, past, hold, features):
        self.dt = np.asarray(dt)
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.t = np.asarray(times)
        self.past = past
        self.hold = hold
        self.x = np.column_stack((np.ones(len(hold)), features))

    def design(self, log_taus):
        tau = np.exp(log_taus)
        u, a, t = (z[:, None] / tau for z in (self.dt, self.after, self.t))
        ed, ea, et = np.exp(-u), np.exp(-a), np.exp(-t)
        step, response = -np.expm1(-u), -np.expm1(-t)
        gap = self.hold[:, None] - self.past @ (ea * step)
        dgap = -self.past @ (ea * (a * step - u * ed))
        n, k = len(gap), len(self.t)
        design = np.empty((n, k + 1, 7))
        deriv = np.empty((n, k + 1, 2))
        design[:, :-1, :2] = gap[:, None, :] * response
        design[:, -1, :2] = gap / tau
        design[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        design[:, -1, 2:] = self.x
        deriv[:, :-1] = dgap[:, None, :] * response - gap[:, None, :] * (t * et)
        deriv[:, -1] = (dgap - gap) / tau
        return design, deriv


def _mad(values):
    return max(1.4826 * np.median(np.abs(values - np.median(values))), 1e-8)


def _fit(model, y):
    cache = [None, None, None]

    def evaluate(theta, scale):
        if cache[0] is None or not np.array_equal(theta[:2], cache[0]):
            cache[:] = [theta[:2].copy(), *model.design(theta[:2])]
        design, deriv = cache[1:]
        return (design @ theta[2:] - y) / scale, design, deriv

    def fun(theta, scale):
        return evaluate(theta, scale)[0].ravel()

    def jac(theta, scale):
        _, design, deriv = evaluate(theta, scale)
        j = np.concatenate((deriv * theta[2:4], design), axis=2)
        return (j / np.asarray(scale)[..., None]).reshape(-1, 9)

    t_min, t_max = model.t[model.t > 0].min(), model.t.max()
    low, high = np.log(t_min / 25), np.log(t_max * 20)
    bounds = (np.r_[low, low, np.full(7, -np.inf)],
              np.r_[high, high, np.full(7, np.inf)])
    seeds = np.log(t_max * np.array([[.07, .5], [.025, .3], [.14, 1.5], [.3, 2.5]]))
    p = seeds[0]
    design, _ = model.design(p)
    beta = np.linalg.lstsq(design.reshape(-1, 7), y.ravel(), rcond=None)[0]
    r = design @ beta - y
    scale = np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]
    warm = least_squares(fun, np.r_[p, beta], jac=jac, args=(scale,), bounds=bounds,
                         loss='soft_l1', max_nfev=120).x
    r = (fun(warm, scale).reshape(y.shape) * scale)
    scale = 1.25 * np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]
    candidates = [warm]
    for p in seeds:
        design, _ = model.design(p)
        weighted = design / scale[:, None]
        beta = np.linalg.lstsq(weighted.reshape(-1, 7), (y / scale).ravel(), rcond=None)[0]
        candidates.append(np.r_[p, beta])
    best = min((least_squares(fun, p, jac=jac, args=(scale,), bounds=bounds,
                              loss='cauchy', max_nfev=150) for p in candidates), key=lambda r: r.cost)
    taus, weights = np.exp(best.x[:2]), best.x[2:].copy()
    order = np.argsort(taus)
    weights[:2] = weights[:2][order]
    return taus[order], weights


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    past, hold, x, strain0 = _inputs(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    t, dt = dataset['eval_times'], dataset['past_dt']
    y = np.column_stack((train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs']))
    taus, weights = _fit(_Model(dt, t, past[:n], hold[:n], x[:n]), y)
    design, _ = _Model(dt, t, past[n:], hold[n:], x[n:]).design(np.log(taus))
    pred = design @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': strain0[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
