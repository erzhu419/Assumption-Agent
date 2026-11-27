"""Calibrated two-branch Prony model for amorphous linear flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_line(records, channel):
    if isinstance(records, list):
        for record in reversed(records):
            line = _historical_line(record, channel)
            if line is not None:
                return line
    elif isinstance(records, dict):
        if {'lab_id', 'cmd_stress', 'true_stress'} <= records.keys():
            lab, cmd, true = (np.asarray(records[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
            use = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
            x, y = cmd[use], true[use]
            clean = _clean_sweep(x, y)
            if np.unique(x[clean]).size >= 2:
                return _line(x[clean], y[clean])
    return None


def _clean_sweep(x, y):
    if np.unique(x).size < 2:
        return np.ones(x.size, dtype=bool)
    i, j = np.triu_indices(x.size, 1)
    distinct = x[j] != x[i]
    i, j = i[distinct], j[distinct]
    slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
    return np.abs(y - slope * x - np.median(y - slope * x)) <= 3.0


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(cal):
    lines = np.empty((4, 2))
    history = None
    lab, cmd, true = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    for c in range(4):
        take = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        clean = _clean_sweep(x, y)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or clean.all()):
            lines[c] = _line(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            line = _historical_line(history, c)
            if line is None:
                raise ValueError(f'No usable calibration for channel {c}')
            lines[c] = line
    return lines


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    return np.where(np.isfinite(aligned), aligned, fills[ids])


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.0]
    u = remaining[:, None] / tau
    decay = np.exp(-u)
    gap = hold[:, None] - stress @ np.diff(decay, axis=0)
    dgap = -stress @ np.diff(u * decay, axis=0)
    v = times[:, None] / tau
    prony = -np.expm1(-v)
    a = np.empty((len(hold), len(times) + 1, 7))
    a[:, :-1, :2] = gap[:, None, :] * prony
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:] = xd[:, None, :] * times[None, :, None]
    a[:, -1, 2:] = xd
    deriv = np.empty(a.shape[:-1] + (2,))
    deriv[:, :-1] = dgap[:, None, :] * prony - gap[:, None, :] * (v * np.exp(-v))
    deriv[:, -1] = (dgap - gap) / tau
    return a, deriv


def _fit(inputs, dt, times, y):
    observed = np.isfinite(y)
    # A second divided difference cancels the terminal flow, including any
    # coupon-dependent drift. Its MAD gives an outlier-resistant noise scale.
    k = len(times)
    a0 = (times[1:-1] - times[:-2]) / (times[2:] - times[:-2])
    d2 = y[:, 1:k-1] - (1 - a0) * y[:, :k-2] - a0 * y[:, 2:k]
    d2 -= np.nanmedian(d2, axis=0)
    curve_scale = max(1.4826 * np.nanmedian(np.abs(d2) / np.sqrt(1 + a0**2 + (1-a0)**2)), 1e-8)
    scale = np.full(y.shape[1], curve_scale)
    bounds = (np.r_[np.log([0.01, 0.01]), np.zeros(2), np.full(5, -np.inf)],
              np.r_[np.log([100., 100.]), np.full(7, np.inf)])

    def run(p, width):
        def fun(p):
            a, _ = _design(p[:2], inputs, dt, times)
            return ((a @ p[2:] - y) / scale)[observed]

        def jac(p):
            a, da = _design(p[:2], inputs, dt, times)
            return (np.concatenate((da * p[2:4], a), axis=-1) / scale[None, :, None])[observed]

        return least_squares(fun, p, jac=jac, bounds=bounds, loss='cauchy', f_scale=width,
                             x_scale='jac', max_nfev=120)

    trials = []
    for taus in ((0.25, 2.), (0.55, 4.), (0.9, 8.), (0.15, 12.)):
        lt = np.log(taus)
        a, _ = _design(lt, inputs, dt, times)
        # Rate measurements are far less prone to impulses than strain.
        rate = observed[:, -1]
        w = np.linalg.lstsq(a[rate, -1], y[rate, -1], rcond=None)[0]
        w[:2] = np.maximum(w[:2], 1e-5)
        p = run(np.r_[lt, w], 10.).x
        p = run(p, 4.).x
        trials.append(run(p, 2.5))
    p = min(trials, key=lambda result: result.cost).x
    residual = _design(p[:2], inputs, dt, times)[0] @ p[2:] - y
    scale[:] = np.maximum(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual, axis=0)), axis=0),
                          curve_scale * 0.35)
    p = run(p, 2.5).x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    lines = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    n = len(train['lab_id'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    # Imputed predictors are needed at inference, but measured responses
    # on these rows still contain the unknown true coupon effects.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    inputs = _inputs(train, aligned[:n], lines)
    lt, weights = _fit(tuple(v[complete] for v in inputs), dt, times, y[complete])
    prediction = _design(lt, _inputs(test, aligned[n:], lines), dt, times)[0] @ weights
    return {'estimated_taus': np.exp(lt), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + prediction[:, :-1],
            'initial_rate_pred': prediction[:, -1]}
