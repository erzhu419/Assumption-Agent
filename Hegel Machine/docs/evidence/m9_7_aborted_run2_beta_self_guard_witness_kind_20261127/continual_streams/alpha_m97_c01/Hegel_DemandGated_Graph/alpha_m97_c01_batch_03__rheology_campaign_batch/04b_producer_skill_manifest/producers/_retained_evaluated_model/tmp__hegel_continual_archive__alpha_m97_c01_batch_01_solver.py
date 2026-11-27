"""Calibrated two-branch creep with a damage-gated crystallization plateau."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    labs = np.asarray(rows['lab_id'])
    cmd, true = (np.asarray(rows[k], float) for k in ('cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        m = (labs == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[m], true[m]
        if 2 <= len(x) <= 3:
            # A sparse affine fit can absorb a spike entirely. In this
            # campaign clean command-to-true corrections are below half
            # the EMI cutoff, which separates a single spike from its peers.
            suspect = np.abs(y - x) > 1.5
            if suspect.sum() == 1:
                x, y = x[~suspect], y[~suspect]
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            keep = x[i] != x[j]
            i, j = i[keep], j[keep]
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            errs = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
            if len(errs):
                score = np.median(errs, axis=1)
                best = np.argmin(score)
                core = errs[best] <= max(8 * score[best], 1e-10)
                a, b = _line(x[core], y[core])
                clean = np.abs(a * x + b - y) <= 3.0
                x, y = x[clean], y[clean]
        if len(x) < 2:
            if history is None:
                path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
                with path.open() as handle:
                    history = json.load(handle)
            for sweep in reversed(history):
                m = np.asarray(sweep['lab_id']) == lab
                hx, hy = (np.asarray(sweep[k])[m] for k in ('cmd_stress', 'true_stress'))
                if len(hx) >= 2:
                    x, y = hx, hy
                    break
        result[lab] = _line(x, y)
    return result


def _inputs(dataset):
    def stack(key):
        return np.concatenate([dataset[part][key] for part in ('train', 'test')])

    lab, stratum = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    x = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel
        center = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        x[m] = (raw[m] - center) / scale
    means = []
    for channel, group in np.unique(np.column_stack((lab, stratum)), axis=0):
        v = x[(lab == channel) & (stratum == group)]
        count = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, x, stack('strain_0')


class _Model:
    def __init__(self, dataset, past, hold, x):
        self.dt = np.asarray(dataset['past_dt'])
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.]
        self.t = np.asarray(dataset['eval_times'])
        self.past, self.hold = past, hold
        self.x = np.column_stack((np.ones(len(x)), x))
        damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.) ** 2 @ self.dt
        self.active = (dataset['shear_banding_index'] <= .50) & (damage > 0)
        self.act = np.where(self.active, np.tanh(damage), 0.)
        # Shear selects the phase clock; specimen damage gates only the extra memory terms.
        clock = np.log1p(self.t) if dataset['shear_banding_index'] <= .50 else self.t
        phi = np.broadcast_to(clock, (len(x), len(self.t)))
        self.terminal = np.concatenate((phi, np.ones((len(x), 1))), axis=1)
        self.drift_basis = self.terminal[:, :, None] * self.x[:, None, :]

    def design(self, theta, plateau=True):
        tau = np.exp(theta)
        u, a, t = (v[:, None] / tau for v in (self.dt, self.after, self.t))
        step, decay = -np.expm1(-u), np.exp(-a)
        z = self.past @ (decay * step)
        dz = self.past @ (decay * (a * step - u * np.exp(-u)))
        gap = self.hold[:, None] - z
        h = np.vstack((-np.expm1(-t), 1 / tau))
        dh = np.vstack((-t * np.exp(-t), -1 / tau))
        d = np.zeros((len(z), len(h), 9))
        dp = np.zeros((len(z), len(h), 4))
        d[:, :, :2] = gap[:, None, :] * h
        dp[:, :, :2] = gap[:, None, :] * dh - dz[:, None, :] * h
        if plateau:
            d[:, :, 2:7] = self.drift_basis
            d[:, :, 7] = (self.act * self.hold)[:, None] * h[:, 0]
            d[:, :, 8] = (self.act * z[:, 0])[:, None] * self.terminal
            dp[:, :, 2] = (self.act * self.hold)[:, None] * dh[:, 0]
            dp[:, :, 3] = (self.act * dz[:, 0])[:, None] * self.terminal
        else:
            d[:, :, 2:7] = np.r_[self.t, 1][None, :, None] * self.x[:, None, :]
        return d, dp


def _fit(model, y, plateau=True):
    count = 9 if plateau and np.any(model.active) else 7
    cached = None
    d = dp = None
    scale = 1.

    def evaluate(theta):
        nonlocal cached, d, dp
        if cached is None or not np.array_equal(cached, theta[:2]):
            cached = theta[:2].copy()
            d, dp = model.design(cached, plateau)
            d = d[:, :, :count]
        return d

    def residual(theta):
        return ((evaluate(theta) @ theta[2:] - y) / scale).ravel()

    def jacobian(theta):
        design = evaluate(theta)
        j_tau = dp[:, :, :2] * theta[2:4]
        if count == 9:
            j_tau[:, :, 0] += dp[:, :, 2:] @ theta[9:11]
        return (np.concatenate((j_tau, design), axis=2) /
                np.asarray(scale)[..., None]).reshape(-1, count + 2)

    tmax = model.t.max()
    low, high = model.t[model.t > 0].min() / 25, 20 * tmax
    bounds = (np.r_[np.log([low, low]), np.full(count, -np.inf)],
              np.r_[np.log([high, high]), np.full(count, np.inf)])
    seeds = np.log(tmax * np.array([[.065, .42], [.025, .28], [.15, 1.1], [.3, 2.5], [.42, .065]]))
    best = None
    for seed in seeds:
        w0 = np.linalg.lstsq(evaluate(seed).reshape(-1, count), y.ravel(), rcond=None)[0]
        fit = least_squares(residual, np.r_[seed, w0], jac=jacobian, bounds=bounds,
                            loss='soft_l1', f_scale=.015, x_scale='jac', max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    err = (evaluate(best.x) @ best.x[2:] - y)
    center = np.median(err, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(err - center), axis=0), 1e-7)[None, :]
    fit = least_squares(residual, best.x, jac=jacobian, bounds=bounds,
                        loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    tau = np.exp(fit.x[:2])
    weights = np.zeros(9)
    weights[:count] = fit.x[2:]
    if count == 7 and tau[0] > tau[1]:
        tau = tau[::-1]
        weights[:2] = weights[1::-1]
    return tau, weights


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train = dataset['train']
    n = len(train['lab_id'])
    past, hold, x, strain0 = _inputs(dataset)
    target = np.column_stack((train['strain_curves_obs'] - strain0[:n, None],
                              train['instant_rate_obs']))
    tau, w = _fit(_Model(dataset, past[:n], hold[:n], x[:n]), target)
    design, _ = _Model(dataset, past[n:], hold[n:], x[n:]).design(np.log(tau))
    pred = design @ w
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': strain0[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
