"""Two-branch hereditary creep with a damage-gated crystallization plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    coefficients = np.empty((4, 2))
    history = None
    for channel in range(4):
        selected = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[selected], true[selected]
        emi = False
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            if i.size:
                slopes = (y[j] - y[i]) / (x[j] - x[i])
                errors = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
                best = np.lexsort((np.sum(np.minimum(errors, 3.0), axis=1), np.median(errors, axis=1)))[0]
                core = errors[best] <= max(8 * np.median(errors[best]), 1e-10)
                gain, bias = _affine(x[core], y[core])
                residual = np.abs(gain * x + bias - y)
                emi = bool(np.any(residual > 3.0))
                x, y = x[residual <= 3.0], y[residual <= 3.0]
        if len(x) >= 2 and np.ptp(x) > 0:
            coefficients[channel] = _affine(x, y)
        elif (selected.sum() < 2 or emi) and _HISTORY.is_file():
            if history is None:
                history = json.loads(_HISTORY.read_text())
            for sweep in reversed(history):
                use = np.asarray(sweep['lab_id']) == channel
                hx, hy = (np.asarray(sweep[k])[use] for k in ('cmd_stress', 'true_stress'))
                if len(hx) >= 2 and np.ptp(hx) > 0:
                    gain, bias = _affine(hx, hy)
                    coefficients[channel] = gain, np.median(y - gain * x) if x.size else bias
                    break
            else:
                coefficients[channel] = 1.0, np.median(y - x) if x.size else 0.0
        else:
            coefficients[channel] = 1.0, np.median(y - x) if x.size else 0.0
    return coefficients


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
    def __init__(self, dt, t, past, hold, x, act, plateau_mode):
        self.dt = np.asarray(dt)
        self.age = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.t = np.asarray(t)
        self.past, self.hold, self.act = past, hold, act
        self.regime = bool(np.any(act))
        self.xt = np.column_stack((np.ones(len(x)), x))
        # The batch phase selects the drift kernel; coupon damage gates only the added terms.
        phi = np.broadcast_to(np.log1p(self.t) if plateau_mode else self.t, (len(x), len(self.t)))
        self.drift = self.xt[:, None, :] * np.column_stack((phi, np.ones(len(x))))[:, :, None]
        self.plateau = np.r_[np.log1p(self.t), 1.0]

    def design(self, logtau):
        tau = np.exp(logtau)
        u = self.dt[:, None] / tau
        a = self.age[:, None] / tau
        v = self.t[:, None] / tau
        decay, retention = np.exp(-u), np.exp(-a)
        step = -np.expm1(-u)
        z = self.past @ (retention * step)
        dz = self.past @ (retention * (a * step - u * decay))
        gap = self.hold[:, None] - z
        response = np.vstack((-np.expm1(-v), 1.0 / tau))
        dresponse = np.vstack((-v * np.exp(-v), -1.0 / tau))
        d = np.empty((*self.drift.shape[:2], 9 if self.regime else 7))
        d[:, :, :2] = gap[:, None, :] * response
        d[:, :, 2:7] = self.drift
        prony_grad = gap[:, None, :] * dresponse - dz[:, None, :] * response
        if self.regime:
            d[:, :, 7] = (self.act * self.hold)[:, None] * response[:, 0]
            d[:, :, 8] = (self.act * z[:, 0])[:, None] * self.plateau
            extra = np.stack(((self.act * self.hold)[:, None] * dresponse[:, 0],
                              (self.act * dz[:, 0])[:, None] * self.plateau), axis=-1)
        else:
            extra = None
        return d, prony_grad, extra


def _mad(r):
    return max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-8)


def _fit(model, y):
    k = 9 if model.regime else 7
    cache = [None, None]

    def components(theta):
        if cache[0] is None or not np.array_equal(cache[0], theta[:2]):
            cache[:] = [theta[:2].copy(), model.design(theta[:2])]
        return cache[1]

    def residual(theta, scale):
        return ((components(theta)[0] @ theta[2:] - y) / scale).ravel()

    def jac(theta, scale):
        design, grad, extra = components(theta)
        tau_grad = grad * theta[2:4]
        if extra is not None:
            tau_grad[:, :, 0] += extra @ theta[-2:]
        return (np.concatenate((tau_grad, design), axis=2) / np.asarray(scale)[..., None]).reshape(-1, k + 2)

    tmax = max(model.t.max(), model.dt.max())
    low, high = np.log(min(model.dt.min(), model.t[model.t > 0].min()) / 25), np.log(tmax * 20)
    lb = np.r_[low, low, np.full(k, -np.inf)]
    ub = np.r_[high, high, np.full(k, np.inf)]
    if model.regime:
        # The first branch carries the regime terms; it cannot be relabelled.
        seeds = tmax * np.array([[.09, .48], [.48, .09], [.035, .29], [.29, .035], [.19, 1.3], [1.3, .19]])
    else:
        seeds = tmax * np.array([[.09, .48], [.035, .29], [.19, 1.3]])

    def linear_seed(p, scale):
        design = model.design(p)[0]
        beta = np.linalg.lstsq((design / np.asarray(scale)[..., None]).reshape(-1, k), (y / scale).ravel(), rcond=None)[0]
        return np.r_[p, beta]

    def scales(r):
        return np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]

    seeds = np.log(seeds)
    seed = linear_seed(seeds[0], 1.0)
    scale = scales(residual(seed, 1.0).reshape(y.shape))
    warm = [least_squares(residual, linear_seed(p, scale), jac=jac, args=(scale,),
                          bounds=(lb, ub), loss='soft_l1', max_nfev=100, x_scale='jac') for p in seeds[:2]]
    best_warm = min(warm, key=lambda r: r.cost).x
    scale = 1.25 * scales(residual(best_warm, 1.0).reshape(y.shape))
    starts = [best_warm] + [linear_seed(p, scale) for p in seeds]
    best = min((least_squares(residual, p, jac=jac, args=(scale,), bounds=(lb, ub),
                              loss='cauchy', max_nfev=100, x_scale='jac') for p in starts), key=lambda r: r.cost).x
    logtau, hw = best[:2].copy(), best[2:9].copy()
    rw = best[9:].copy() if model.regime else np.zeros(2)
    if not model.regime:
        order = np.argsort(logtau)
        logtau = logtau[order]
        hw[:2] = hw[:2][order]
    return logtau, hw, rw


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    past, hold, x, strain0 = _inputs(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    act = np.zeros(len(past))
    plateau_mode = dataset['shear_banding_index'] <= 0.50
    if plateau_mode:
        excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0)
        act = np.tanh(excess**2 @ dt)
    y = np.column_stack((train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs']))
    logtau, hw, rw = _fit(_Model(dt, t, past[:n], hold[:n], x[:n], act[:n], plateau_mode), y)
    test = _Model(dt, t, past[n:], hold[n:], x[n:], act[n:], plateau_mode)
    pred = test.design(logtau)[0] @ (np.r_[hw, rw] if test.regime else hw)
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': hw,
            'regime_mode_weights': rw, 'predicted_strain': strain0[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
