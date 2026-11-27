"""Two-branch rheology inversion for the jammed-slip campaign."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(dataset):
    c = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    fits = np.empty((4, 2))
    history = None

    def archived(channel):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        for sweep in reversed(history):
            x, y = (np.asarray(sweep[k]) for k in ('cmd_stress', 'true_stress'))
            rows = (np.asarray(sweep['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
            if rows.sum() >= 2 and np.ptp(x[rows]) > 0:
                return _line(x[rows], y[rows])
        raise ValueError(f'No usable archived sweep for channel {channel}')

    for channel in range(4):
        rows = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[rows], true[rows]
        emi = False
        if len(x) >= 3:
            for i in range(len(x)):
                other = np.arange(len(x)) != i
                if np.ptp(x[other]) > 0:
                    a, b = _line(x[other], y[other])
                    if abs(y[i] - a*x[i] - b) > 3.0:
                        emi = True
                        break
        if emi or len(x) < 2:
            a, b = archived(channel)
            clean = np.abs(y - a*x - b) <= 3.0
            x, y = x[clean], y[clean]
            if len(x) < 2 or np.ptp(x) == 0:
                fits[channel] = a, np.mean(y - a*x) if len(x) else b
                continue
        fits[channel] = _line(x, y)
    return fits


def _column_mean(a):
    count = np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a, axis=0), count,
                     out=np.full(a.shape[1], np.nan), where=count > 0)


def _aligned(train, test):
    def join(key):
        return np.concatenate((train[key], test[key]))
    lab, group, ref = (join(k) for k in ('lab_id', 'stratum', 'ref_coupons'))
    x = join('coupon_features_raw').astype(float)
    cells = {}
    for l in np.unique(lab):
        rows = lab == l
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        x[rows] = (x[rows] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0))
        for s in np.unique(group[rows]):
            cell = rows & (group == s)
            cells[l, s] = cell, _column_mean(x[cell])
    overall = _column_mean(np.array([m for _, m in cells.values()]))
    for l in np.unique(lab):
        local = _column_mean(np.array([m for (ll, _), (_, m) in cells.items() if ll == l]))
        for (ll, _), (rows, mean) in cells.items():
            if ll == l:
                fill = np.where(np.isfinite(mean), mean, np.where(np.isfinite(local), local, overall))
                x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _inputs(split, aligned, calibration):
    gain, bias = calibration[split['lab_id']].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    return past, hold, np.column_stack((np.ones(len(hold)), aligned, slip))


def _design(tau, dt, times, inputs):
    past, hold, flow = inputs
    # Each increment decays for the duration of all subsequent intervals.
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory_kernel = -np.expm1(-dt[:, None] / tau) * np.exp(-after[:, None] / tau)
    gap = hold[:, None] - past @ memory_kernel
    contrast = gap[:, 0] - gap[:, 1]
    saturation = -np.expm1(-times[:, None] / tau)
    terminal = times / (1.0 + 0.85 * times)
    n, k = len(hold), len(times)
    curve = np.empty((n, k, 9))
    curve[:, :, :2] = gap[:, None, :] * saturation
    curve[:, :, 2:8] = flow[:, None, :] * terminal[None, :, None]
    curve[:, :, 8] = contrast[:, None] * saturation[:, 0]
    rate = np.column_stack((gap / tau, flow, contrast / tau[0]))
    return curve, rate


def _fit(dt, times, inputs, curves, rates):
    n, k = curves.shape
    target = np.r_[curves.ravel(), rates]
    split = n * k
    low, high = np.log([0.02, 120.0])

    def matrix(log_tau):
        a, b = _design(np.exp(log_tau), dt, times, inputs)
        return np.concatenate((a.reshape(-1, 9), b))

    def regression(a, root_weight):
        return np.linalg.lstsq(a * root_weight[:, None], target * root_weight, rcond=None)[0]

    def error(a, coef):
        return np.einsum('ij,j->i', a, coef) - target

    def scales(r):
        def mad(v):
            return max(1e-6, 1.4826 * np.median(np.abs(v - np.median(v))))
        return np.r_[np.full(split, mad(r[:split])), np.full(n, mad(r[split:]))]

    # Use an ordinary fit only to initialize the observation noise scales.
    start = np.log([0.55, 4.0])
    a = matrix(start)
    sigma = scales(error(a, regression(a, np.ones(len(target)))))

    def profile(log_tau, sigma):
        a = matrix(log_tau)
        coef = regression(a, 1 / sigma)
        for _ in range(20):
            r = error(a, coef) / sigma
            root_weight = 1 / (sigma * np.sqrt(1 + (r / 2.5)**2))
            nxt = regression(a, root_weight)
            if np.linalg.norm(nxt - coef) <= 1e-8 * (1 + np.linalg.norm(coef)):
                coef = nxt
                break
            coef = nxt
        r = error(a, coef) / sigma
        # Signed square root gives least_squares the Cauchy objective.
        return np.sign(r) * 2.5 * np.sqrt(np.log1p((r / 2.5)**2)), coef, r * sigma

    def optimize(start, sigma):
        return least_squares(lambda p: profile(p, sigma)[0], start,
                             bounds=(low, high), max_nfev=70, diff_step=1e-5,
                             ftol=1e-8, xtol=1e-8, gtol=1e-8)

    starts = [(0.22, 1.4), (0.55, 4.0), (0.95, 8.0), (0.22, 10.0)]
    results = [optimize(np.log(pair), sigma) for pair in starts]
    best = min((r for r in results if r.x[0] < r.x[1]), key=lambda r: r.cost)
    # Re-estimate each instrument's noise scale after discounting EMI spikes.
    sigma = scales(profile(best.x, sigma)[2])
    refined = optimize(best.x, sigma)
    optimum = refined.x if refined.x[0] < refined.x[1] else best.x
    return np.exp(optimum), profile(optimum, sigma)[1]


def solve_alpha_m97_c02_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    calibration = _calibration(dataset)
    x_train, x_test = _aligned(train, test)
    # Incomplete coupons carry extra uncertainty in the shared drift term.
    # Use measured coupons to estimate material parameters, and the prescribed
    # panel imputations when predicting those coupons.
    measured = np.isfinite(train['coupon_features_raw']).all(axis=1)
    inputs = _inputs(train, x_train, calibration)
    tau, weights = _fit(dt, times, tuple(a[measured] for a in inputs),
                        (train['strain_curves_obs'] - train['strain_0'][:, None])[measured],
                        train['instant_rate_obs'][measured])
    curve, rate = _design(tau, dt, times, _inputs(test, x_test, calibration))
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
