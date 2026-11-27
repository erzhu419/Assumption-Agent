"""Load-cell calibrated, two-branch jammed-slip creep solver."""
import json

import numpy as np
from scipy.optimize import least_squares


def _design(logtau, past, hold, features, dt, times):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    decay = np.exp(-dt[:, None] / tau)
    increment = -np.expm1(-dt[:, None] / tau)
    for s, (a, b) in enumerate(zip(decay, increment)):
        z = a*z + b*past[:, s, None]
    gap = hold[:, None] - z
    gap_diff = gap[:, 0] - gap[:, 1]
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    responses = -np.expm1(-times[:, None]/tau)
    terminal = times/(1 + 0.85*times)
    drift_slip = np.column_stack((np.ones(len(hold)), features, slip))
    curve = np.concatenate((gap[:, None, :]*responses[None, :, :],
                            drift_slip[:, None, :]*terminal[None, :, None],
                            (gap_diff[:, None]*responses[None, :, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, drift_slip, gap_diff/tau[0]))
    return curve, rates


def _fit(past, hold, features, dt, times, strain, rate):
    use_curve = np.isfinite(strain)
    use_rate = np.isfinite(rate)
    y = np.r_[strain[use_curve], rate[use_rate]]
    ncurve = np.count_nonzero(use_curve)
    bounds = np.log([[0.025, 0.025], [60., 60.]])

    def design(p):
        curves, rates = _design(p, past, hold, features, dt, times)
        return np.concatenate((curves[use_curve], rates[use_rate]))

    def project(p, weights, robust=False):
        a = design(p)
        w = weights
        for _ in range(7 if robust else 1):
            coef = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
            e = a@coef-y
            if robust:
                w = weights*(1 + (e*weights/2.5)**2)**(-0.25)
        if robust:
            u = e*weights/2.5
            return (e*weights)*np.sqrt(2/(np.hypot(1, u)+1)), coef
        return e, coef

    plain = np.ones_like(y)
    def objective(p):
        return project(p, plain)[0]
    fits = [least_squares(objective, np.log(start), bounds=bounds, diff_step=1e-4)
            for start in ((0.4, 3.), (1., 8.), (0.2, 1.5))]
    best = min((f for f in fits if f.x[0] < f.x[1]), key=lambda f: f.cost)
    residual = objective(best.x)
    def scale(group):
        return max(1e-6, 1.4826*np.median(np.abs(group - np.median(group)))) if len(group) else 1.
    weights = np.r_[np.full(ncurve, 1/scale(residual[:ncurve])),
                    np.full(len(y)-ncurve, 1/scale(residual[ncurve:]))]
    result = least_squares(lambda p: project(p, weights, True)[0], best.x,
                           bounds=bounds, diff_step=1e-4, max_nfev=100)
    p = result.x if result.x[0] < result.x[1] else best.x
    return p, project(p, weights, True)[1]

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _prior_calibration(lab, x, y):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    candidates = []
    for sweep in reversed(history):
        ids, cmd, true = (np.asarray(sweep[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
        rows = (ids == lab) & np.isfinite(cmd) & np.isfinite(true)
        if rows.sum() >= 2 and np.ptp(cmd[rows]) > 0:
            candidates.append(np.polyfit(cmd[rows], true[rows], 1))
    if not len(x):
        return candidates[0]

    def support(fit):
        error = np.abs(y - np.polyval(fit, x))
        clean = error <= 0.01
        return -np.count_nonzero(clean), np.count_nonzero((error <= 3) & ~clean)

    prior = min(candidates, key=support)
    clean = np.abs(y - np.polyval(prior, x)) <= 0.01
    if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
        return np.polyfit(x[clean], y[clean], 1)
    return prior


def _calibration(table):
    ids, cmd, true = (np.asarray(table[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    fits = []
    for lab in range(4):
        rows = (ids == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[rows], true[rows]
        if len(x) < 2:
            fits.append(_prior_calibration(lab, x, y))
            continue
        fit = np.polyfit(x, y, 1)
        if np.max(np.abs(y - np.polyval(fit, x))) > 0.01:
            consensus = []
            for i, j in zip(*np.triu_indices(len(x), 1)):
                if x[i] == x[j]:
                    continue
                a = (y[j] - y[i])/(x[j] - x[i])
                error = np.abs(y - y[i] - a*(x - x[i]))
                clean = error <= 0.01
                if np.any(error > 3.) and np.all(clean | (error > 3.)):
                    consensus.append(clean)
            if consensus:
                clean = max(consensus, key=np.count_nonzero)
                if clean.sum() >= 4:
                    fit = np.polyfit(x[clean], y[clean], 1)
                else:
                    fit = _prior_calibration(lab, x, y)
        fits.append(fit)
    return np.asarray(fits)
def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]))

    lab, stratum = join('lab_id'), join('stratum')
    ref, features = join('ref_coupons'), join('coupon_features_raw').astype(float)
    cells = []
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[rows] = (features[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            cell = features[rows & (stratum == s)]
            count = np.isfinite(cell).sum(axis=0)
            cells.append(np.divide(np.nansum(cell, axis=0), count,
                                   out=np.full(features.shape[1], np.nan), where=count > 0))
    features = np.where(np.isnan(features), np.nanmean(cells, axis=0), features)
    gain, bias = _calibration(dataset['calibration_load_cells'])[lab].T
    past = join('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    return past, hold, features, join('strain_0'), np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])




def solve_alpha_m97_c03_batch_06(dataset: dict) -> dict:
    past, hold, features, offset, dt, times = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    # Fit on measured coupons; use cell-balanced imputations for prediction.
    measured = np.isfinite(train['coupon_features_raw']).all(axis=1)
    args = past[:n][measured], hold[:n][measured], features[:n][measured], dt, times
    strain = (train['strain_curves_obs'] - offset[:n, None])[measured]
    rate = train['instant_rate_obs'][measured]
    logtau, weights = _fit(*args, strain, rate)
    # Isolated strain spikes do not invalidate the other times or the rate.
    train_curves, _ = _design(logtau, *args)
    residual = train_curves @ weights - strain
    center = np.nanmedian(residual)
    scale = max(1e-6, 1.4826 * np.nanmedian(np.abs(residual - center)))
    spike = np.abs(residual - center) > 6 * scale
    if spike.any():
        logtau, weights = _fit(*args, np.where(spike, np.nan, strain), rate)
    curves, rates = _design(logtau, past[n:], hold[n:], features[n:], dt, times)
    return {
        'estimated_taus': np.exp(logtau),
        'hereditary_weights': weights[:7],
        'regime_mode_weights': weights[7:],
        'predicted_strain': offset[n:, None] + curves @ weights,
        'initial_rate_pred': rates @ weights,
    }
