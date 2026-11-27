"""Fit the calibrated two-branch jammed-slip rheology law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_line(lab):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        m = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[m]
        y = np.asarray(sweep['true_stress'])[m]
        if len(x) >= 2 and np.ptp(x) > 0:
            return np.polyfit(x, y, 1)
    raise ValueError(f'No calibration line for lab {lab}')


def _calibrate(cal):
    lab, x, y = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    lines = []
    for c in range(4):
        m = (lab == c) & np.isfinite(x) & np.isfinite(y)
        u, v = x[m], y[m]
        if 3 <= len(u) <= 4 and np.ptp(u) > 0:
            # A high-leverage endpoint can hide a spike in a short sweep.
            # Check whether omitting a point exposes an EMI-sized deviation.
            fallback = None
            for i in range(len(u)):
                other = np.arange(len(u)) != i
                if np.ptp(u[other]) == 0:
                    continue
                line = np.polyfit(u[other], v[other], 1)
                # Account for calibration noise amplified by extrapolation.
                # The archive check below still requires a residual above 3.0.
                x_other = u[other]
                centered = x_other - x_other.mean()
                leverage = 1.0/len(x_other) + (u[i] - x_other.mean())*centered/(centered@centered)
                allowance = 0.001*(1.0 + np.abs(leverage).sum())
                if abs(v[i] - np.polyval(line, u[i])) > 3.0 - allowance:
                    prior = _historical_line(c)
                    delta = np.abs(v - np.polyval(prior, u))
                    if (delta <= 0.05).sum() >= len(u) - 1 and np.any(delta > 3.0):
                        fallback = prior
                    # The archive is used only to corroborate an isolated spike.
                    break
            if fallback is not None:
                lines.append(fallback)
                continue
        if len(u) >= 3 and np.ptp(u) > 0:
            slopes = []
            for ui, vi in zip(u, v):
                different = u != ui
                slopes.append(np.median((v[different] - vi) / (u[different] - ui)))
            g = np.median(slopes)
            b = np.median(v - g*u)
            keep = np.abs(v - (g*u+b)) <= 3.0
            if not np.all(keep):
                u, v = u[keep], v[keep]
                if len(u) < 4:
                    lines.append(_historical_line(c))
                    continue
        if len(u) >= 2 and np.ptp(u) > 0:
            lines.append(np.polyfit(u, v, 1))
        else:
            lines.append(_historical_line(c))
    return np.asarray(lines)


def _prepare(dataset):
    a, b = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((a[key], b[key]), axis=0)
    lab = joined('lab_id')
    panel = joined('ref_coupons')
    features = joined('coupon_features_raw').astype(float)
    for c in range(4):
        m = lab == c
        r = panel[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        features[m] = (features[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    cells, cell = np.unique(np.c_[lab, joined('stratum')], axis=0, return_inverse=True)
    good = np.isfinite(features)
    sums = np.zeros((len(cells), features.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(good, features, 0.))
    np.add.at(counts, cell, good)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / (counts > 0).sum(axis=0)
    features = np.where(good, features, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*joined('past_stress_cmd') + bias[:, None]
    hold = gain*joined('stress_hold_cmd') + bias
    return past, hold, features


def _design(data, dt, t, tau):
    past, hold, x = data
    memory = np.zeros((len(hold), 2))
    for step, stress in zip(dt, past.T):
        u = -step/tau
        memory = np.exp(u)*memory - np.expm1(u)*stress[:, None]
    gap = hold[:, None] - memory
    prony = -np.expm1(-t[:, None]/tau)
    phi = t/(1.0 + 0.85*t)
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    flow = np.column_stack((np.ones(len(hold)), x, slip))
    contrast = gap[:, 0] - gap[:, 1]
    curves = np.concatenate((gap[:, None, :]*prony,
                             flow[:, None, :]*phi[None, :, None],
                             (contrast[:, None]*prony[:, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, flow, contrast/tau[0]))
    return curves, rates


def _fit(data, dt, t, curves, rates):
    cm, rm = np.isfinite(curves), np.isfinite(rates)
    cy, ry = curves[cm], rates[rm]
    y = np.r_[cy, ry]
    def design(log_tau):
        c, r = _design(data, dt, t, np.exp(log_tau))
        return np.vstack((c[cm], r[rm]))
    def profile(log_tau, scale):
        a = design(log_tau)
        w = np.linalg.lstsq(a/scale[:, None], y/scale, rcond=None)[0]
        return (a@w-y)/scale, w
    def mad(r):
        return max(1e-5, 1.4826*np.median(np.abs(r-np.median(r))))
    lo = np.log(min(dt)*0.12)
    hi = np.log(max(t[-1], sum(dt))*12)
    grid = np.geomspace(min(dt)*0.7, max(t[-1], sum(dt))*1.25, 8)
    seeds = [np.log([a, b]) for i, a in enumerate(grid) for b in grid[i+1:]]
    scale = np.r_[np.ones(len(cy)), np.full(len(ry), 2.)]
    def score(seed):
        r, _ = profile(seed, scale)
        return np.dot(r, r)
    seeds.sort(key=score)
    pilot = least_squares(lambda z: profile(z, scale)[0], seeds[0], bounds=(lo, hi))
    err, _ = profile(pilot.x, np.ones_like(y))
    scale = np.r_[np.full(len(cy), mad(err[:len(cy)])), np.full(len(ry), mad(err[len(cy):]))]
    winners = []
    lower = np.r_[[lo, lo], np.full(9, -np.inf)]
    upper = np.r_[[hi, hi], np.full(9, np.inf)]
    for seed in [pilot.x]+seeds[:4]:
        _, w = profile(seed, scale)
        trial = least_squares(lambda z: (design(z[:2])@z[2:]-y)/scale,
                              np.r_[seed, w], bounds=(lower, upper),
                              loss='soft_l1', f_scale=1.3, max_nfev=100)
        if trial.x[0] < trial.x[1]:
            winners.append(trial)
    if not winners:
        return np.exp(seeds[0]), profile(seeds[0], scale)[1]
    fit = min(winners, key=lambda trial: trial.cost).x
    return np.exp(fit[:2]), fit[2:]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tau, w = _fit(tuple(v[:n] for v in data), dt, t,
                  train['strain_curves_obs'] - train['strain_0'][:, None],
                  train['instant_rate_obs'])
    curves, rates = _design(tuple(v[n:] for v in data), dt, t, tau)
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curves@w,
            'initial_rate_pred': rates@w}
