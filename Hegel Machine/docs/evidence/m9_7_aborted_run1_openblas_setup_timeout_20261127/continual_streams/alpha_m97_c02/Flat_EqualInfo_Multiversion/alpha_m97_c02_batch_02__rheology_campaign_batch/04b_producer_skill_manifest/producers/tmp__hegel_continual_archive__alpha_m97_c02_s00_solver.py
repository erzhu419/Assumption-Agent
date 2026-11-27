"""Calibrated two-branch Prony fit for the linear-flow campaign."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    if isinstance(history, dict):
        history = history.get('channels', history.get('calibration_load_cells', history))
    if isinstance(history, dict) and 'lab_id' in history:
        if 'cmd_stress' in history and 'true_stress' in history:
            mask = np.asarray(history['lab_id']) == channel
            x = np.asarray(history['cmd_stress'])[mask]
            y = np.asarray(history['true_stress'])[mask]
            return np.polyfit(x, y, 1) if np.unique(x).size >= 2 else None
    if isinstance(history, dict):
        entry = history.get(str(channel))
    else:
        entry = next((row for row in reversed(history)
                      if isinstance(row, dict) and row.get('lab_id') == channel
                      and 'gain' in row and 'bias' in row), None)
    if entry is not None:
        return (entry['gain'], entry['bias']) if isinstance(entry, dict) else entry
    return None


def _calibrate(table):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        i, j = np.triu_indices(x.size, k=1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        spike = False
        if i.size:
            gain = np.median((y[i] - y[j]) / (x[i] - x[j]))
            offsets = y - gain*x
            clean = np.abs(offsets - np.median(offsets)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
        archive = _archived_calibration(channel) if x.size < 2 or spike else None
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
            result[channel] = (1.0, np.median(y - x) if y.size else 0.0)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, x, ref, past, hold, strain0 = (
        np.concatenate((train[key], test[key])) for key in keys
    )
    x = x.astype(float, copy=False)
    for channel in np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, dt, times):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # Unroll the recurrence from z(0)=0; each history interval's
    # contribution decays through all subsequent intervals.
    memory_weights = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory_weights
    dgap = -past @ (a*memory_weights - q*ea*eq)
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :]*step,
                            times[None, :, None]*features[:, None, :]), axis=2)
    rate = np.column_stack((gap/taus, features))
    dcurve = dgap[:, None, :]*step - gap[:, None, :]*r*er
    drate = (dgap - gap)/taus
    return curve, rate, dcurve, drate


def _fit(past, hold, features, dt, times, curves, rates):
    scales = np.r_[np.full(curves.size, 0.025), np.full(rates.size, 0.025)]
    target = np.r_[curves.ravel(), rates]

    def design(log_taus):
        curve, rate, dcurve, drate = _basis(log_taus, past, hold, features, dt, times)
        return (np.vstack((curve.reshape(-1, 7), rate)),
                np.vstack((dcurve.reshape(-1, 2), drate)))

    def residual(params):
        matrix, _ = design(params[:2])
        return (np.einsum('ij,j->i', matrix, params[2:]) - target) / scales

    def jacobian(params):
        matrix, deriv = design(params[:2])
        return np.column_stack((deriv*params[2:4], matrix)) / scales[:, None]

    # Downweight large glitches for initialization, then minimize the
    # observation-level Cauchy loss on both the curve and its initial slope.
    weight = 1 / np.maximum(1, np.abs(target)/max(np.quantile(np.abs(target), 0.9), 0.01))
    lower = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    upper = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0),
                 (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        matrix, _ = design(log_taus)
        initial = np.linalg.lstsq(matrix*weight[:, None], target*weight, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, initial], jac=jacobian,
                            bounds=(lower, upper), loss='cauchy', x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x
    order = np.argsort(params[:2])
    return params[:2][order], np.r_[params[2:4][order], params[4:]]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    """Estimate the shared rheology and predict every test coupon."""
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[key], dtype=float) for key in ('past_dt', 'eval_times'))
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    log_taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times,
                             curves, np.asarray(train['instant_rate_obs']))
    curve, rate, _, _ = _basis(log_taus, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
