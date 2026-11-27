"""Fit calibrated two-branch polymer rheology with a crystallized plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(lab):
    if not _ARCHIVE.is_file():
        return None
    history = json.loads(_ARCHIVE.read_text())
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[mask]
        y = np.asarray(sweep['true_stress'])[mask]
        if len(x) >= 2 and np.ptp(x) > 0:
            return np.polyfit(x, y, 1)
    return None


def _calibrate(c):
    result = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        mask = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][mask], c['true_stress'][mask]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            # A repeated-median line detects gross calibration errors before OLS.
            dx = x[:, None] - x
            dy = y[:, None] - y
            with np.errstate(divide='ignore', invalid='ignore'):
                slopes = [np.median(yi[xi != 0] / xi[xi != 0]) for xi, yi in zip(dx, dy)]
            gain = np.median(slopes)
            bias = np.median(y - gain*x)
            good = np.abs(y - gain*x - bias) <= 3.0
            spikes = not np.all(good)
            if spikes:
                x, y = x[good], y[good]
        historical = _historical_calibration(lab) if len(x) < 2 or spikes else None
        if len(x) >= 2 and np.ptp(x) > 0 and (len(x) >= 4 or historical is None):
            result[lab] = np.polyfit(x, y, 1)
        elif historical is not None:
            result[lab] = historical
        elif len(x):
            result[lab, 1] = np.median(y - x)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    train_size = len(train['lab_id'])
    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab = join('lab_id')
    strata = join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x = join('coupon_features_raw')
    ref = join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        eligible = np.isfinite(panel).sum(axis=0) >= 2
        panel = panel[:, eligible]
        loc = np.median(np.nanmean(panel, axis=0))
        std = np.median(np.nanstd(panel, axis=0))
        x[sel] = (x[sel] - loc)/std
    # Each cell with an observation contributes one mean per feature.
    cells, cell = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True)
    valid = np.isfinite(x)
    sums = np.zeros((len(cells), x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(valid, x, 0.))
    np.add.at(counts, cell, valid)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    observed_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), observed_cells,
                     out=np.zeros(x.shape[1]), where=observed_cells > 0)
    x = np.where(valid, x, fill)
    return (past, hold, x), train_size


def _state(data, dt, crit, plateau):
    past, hold, x = data
    excess = np.maximum(np.abs(past) - crit, 0.)
    activation = np.tanh((excess**2) @ dt) if plateau else np.zeros_like(hold)
    return past, hold, x, activation


def _design(logtaus, state, dt, times, plateau):
    past, hold, x, activation = state
    tau = np.exp(logtaus)
    z = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        u = -width/tau
        z = np.exp(u)*z - np.expm1(u)*stress[:, None]
    gap = hold[:, None] - z
    transient = -np.expm1(-times[:, None]/tau)
    phi = np.log1p(times) if plateau else times
    drift = np.column_stack((np.ones(len(hold)), x))
    a = np.empty((len(hold), len(times), 9))
    a[:, :, :2] = gap[:, None, :]*transient
    a[:, :, 2:7] = drift[:, None, :]*phi[None, :, None]
    a[:, :, 7] = (activation*hold)[:, None]*transient[:, 0]
    a[:, :, 8] = (activation*z[:, 0])[:, None]*phi
    rate = np.column_stack((gap/tau, drift, activation*hold/tau[0], activation*z[:, 0]))
    return a, rate


def _fit(state, dt, times, plateau, delta, rate_obs=None):
    curve_good = np.isfinite(delta)
    rate_good = np.isfinite(rate_obs) if rate_obs is not None else None
    use_rates = rate_good is not None and rate_good.any()
    def matrices(logtaus):
        a, rate = _design(logtaus, state, dt, times, plateau)
        return a[curve_good], rate[rate_good] if use_rates else None

    def profile(logtaus, rate_weight):
        a, rate = matrices(logtaus)
        y = delta[curve_good]
        if rate_weight > 0:
            a = np.vstack((a, rate_weight*rate))
            y = np.r_[y, rate_weight*rate_obs[rate_good]]
        weights = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, weights) - y, weights

    bounds = np.log([max(np.min(dt)*0.05, 1e-4), 20*(dt.sum()+times.max())])
    starts = ([0.35, 2.5], [0.75, 4.0], [1.5, 7.0])
    if plateau and np.any(state[3]):
        starts += tuple(pair[::-1] for pair in starts)
    def fit_from(start, rate_weight):
        return least_squares(lambda p: profile(p, rate_weight)[0], start,
                             bounds=(bounds[0], bounds[1]), max_nfev=90)

    fits = [fit_from(np.clip(np.log(pair), *bounds), 0.) for pair in starts]
    best = min(fits, key=lambda fit: fit.cost)
    initial, w = profile(best.x, 0.)

    def scale(residual):
        return max(1.4826*np.median(np.abs(residual-np.median(residual))), 1e-8)

    curve_scale = scale(initial)
    _, rate = matrices(best.x)
    if use_rates:
        rate_scale = scale(rate @ w-rate_obs[rate_good])
        rate_weight = curve_scale/rate_scale
        best = min((fit_from(fit.x, rate_weight) for fit in fits), key=lambda fit: fit.cost)
        initial, w = profile(best.x, rate_weight)
        curve_scale = scale(initial[:curve_good.sum()])
        _, rate = matrices(best.x)
        rate_scale = scale(rate @ w-rate_obs[rate_good])
        target = np.r_[delta[curve_good]/curve_scale, rate_obs[rate_good]/rate_scale]
    else:
        target = delta[curve_good]/curve_scale

    def robust(logtaus, return_weights=False):
        a, rate = matrices(logtaus)
        a = a/curve_scale
        if use_rates:
            a = np.vstack((a, rate/rate_scale))
        weights = np.linalg.lstsq(a, target, rcond=None)[0]
        for _ in range(10):
            e = np.einsum('ij,j->i', a, weights)-target
            v = 1/np.hypot(1., e/2.)
            weights = np.linalg.lstsq(a*v[:, None], target*v, rcond=None)[0]
        if return_weights:
            return weights
        e = np.einsum('ij,j->i', a, weights)-target
        return np.sign(e)*2*np.sqrt(np.log1p((e/2.)**2))

    robust_starts = [best.x, np.clip(np.log([0.75, 4.0]), *bounds)]
    if plateau and np.any(state[3]):
        robust_starts.append(best.x[::-1])
    best = min((least_squares(robust, start, bounds=tuple(bounds), max_nfev=90)
                for start in robust_starts), key=lambda fit: fit.cost)
    logs = best.x.copy()
    weights = robust(logs, return_weights=True)
    if not np.any(state[3]) and logs[0] > logs[1]:
        logs = logs[::-1]
        weights[:2] = weights[1::-1]
    return logs, weights


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    dt, times = dataset['past_dt'], dataset['eval_times']
    train, test = dataset['train'], dataset['test']
    data, n = _prepare(dataset)
    plateau = dataset['shear_banding_index'] <= 0.50
    state = _state(data, dt, dataset['crit_stress'], plateau)
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, weights = _fit(tuple(s[:n] for s in state), dt, times, plateau, delta,
                         train.get('instant_rate_obs'))
    a, rate = _design(logs, tuple(s[n:] for s in state), dt, times, plateau)
    return {'estimated_taus': np.exp(logs), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + a @ weights,
            'initial_rate_pred': rate @ weights}
