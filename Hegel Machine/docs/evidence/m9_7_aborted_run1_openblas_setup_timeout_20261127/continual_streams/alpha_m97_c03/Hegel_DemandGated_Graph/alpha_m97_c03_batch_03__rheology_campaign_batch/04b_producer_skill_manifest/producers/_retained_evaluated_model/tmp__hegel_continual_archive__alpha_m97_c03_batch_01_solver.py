import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibration(cal):
    coefficients = np.empty((4, 2))
    history = None
    for lab in range(4):
        mask = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][mask], cal['true_stress'][mask]
        good = np.ones(len(x), dtype=bool)
        if np.unique(x).size >= 2:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            slope = np.median((y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]]))
            intercept = np.median(y - slope*x)
            good = np.abs(y - slope*x - intercept) <= 3.0
        if good.sum() >= 4 or (good.all() and np.unique(x).size >= 2):
            coefficients[lab] = _line(x[good], y[good])
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for sweep in reversed(history):
            hist_lab = np.asarray(sweep['lab_id'])
            a = np.asarray(sweep['cmd_stress'])[hist_lab == lab]
            b = np.asarray(sweep['true_stress'])[hist_lab == lab]
            if np.unique(a).size >= 2:
                coefficients[lab] = _line(a, b)
                break
        else:
            raise ValueError(f'No usable calibration for lab {lab}')
    return coefficients


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in (
        'lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        ref = panel[mask]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - center) / scale
    keys = np.unique(np.column_stack((labs, strata)), axis=0)
    cells = [(labs == lab) & (strata == stratum) for lab, stratum in keys]
    means = []
    for mask in cells:
        v = aligned[mask]
        n = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), n, out=np.full(v.shape[1], np.nan), where=n > 0))
    means = np.asarray(means)
    for i, mask in enumerate(cells):
        fill = means[i].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            local = means[keys[:, 0] == keys[i, 0], j]
            fill[j] = np.nanmean(local) if np.isfinite(local).any() else np.nanmean(means[:, j])
        aligned[mask] = np.where(np.isfinite(aligned[mask]), aligned[mask], fill)
    return aligned


def _inputs(dataset, part, aligned, coefficients):
    gain, bias = coefficients[part['lab_id']].T
    stress = gain[:, None]*part['past_stress_cmd'] + bias[:, None]
    hold = gain*part['stress_hold_cmd'] + bias
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    damage = np.maximum(0., np.abs(stress) - dataset['crit_stress'])**2 @ dt
    plateau = (dataset['shear_banding_index'] <= .50) & (damage > 0.)
    activation = np.where(plateau, np.tanh(damage), 0.)
    return stress, hold, aligned, dt, times, plateau, activation, dataset['shear_banding_index'] <= .50


def _design(logtau, stress, hold, x, dt, times, plateau, activation, crystallization_campaign):
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
    u = remaining[:, None]/tau
    e = np.exp(-u)
    mem = stress @ np.diff(e, axis=0)
    dmem = stress @ np.diff(u*e, axis=0)
    gap = hold[:, None] - mem
    v = times[:, None]/tau
    phi = -np.expm1(-v)
    dphi = -v*np.exp(-v)
    # The clock is shared in this campaign; damage gates the added amplitudes.
    terminal = np.log1p(times) if crystallization_campaign else times
    drift_basis = np.column_stack((np.ones(len(x)), x))
    primary = activation*hold
    secondary = activation*mem[:, 0]
    n, k = len(x), len(times)
    a = np.empty((n, k+1, 9))
    a[:, :k, :2] = gap[:, None, :]*phi
    a[:, k, :2] = gap/tau
    a[:, :k, 2:7] = drift_basis[:, None, :]*terminal[None, :, None]
    a[:, k, 2:7] = drift_basis
    a[:, :k, 7] = primary[:, None]*phi[:, 0]
    a[:, k, 7] = primary/tau[0]
    a[:, :k, 8] = secondary[:, None]*terminal
    a[:, k, 8] = secondary
    d = np.empty((n, k+1, 4))
    d[:, :k, :2] = -dmem[:, None, :]*phi + gap[:, None, :]*dphi
    d[:, k, :2] = (-dmem-gap)/tau
    d[:, :k, 2] = primary[:, None]*dphi[:, 0]
    d[:, k, 2] = -primary/tau[0]
    dsecondary = activation*dmem[:, 0]
    d[:, :k, 3] = dsecondary[:, None]*terminal
    d[:, k, 3] = dsecondary
    return a, d


def _fit(inputs, y):
    dt, times = inputs[3:5]
    low = np.r_[np.full(2, np.log(min(dt.min(), times[times > 0].min())/30)), np.full(9, -np.inf)]
    high = np.r_[np.full(2, np.log((dt.sum()+times.max())*15)), np.full(9, np.inf)]

    def optimize(p, scale, loss):
        def fun(p):
            a, _ = _design(p[:2], *inputs)
            return ((a @ p[2:] - y)/scale).ravel()

        def jac(p):
            a, d = _design(p[:2], *inputs)
            fast = d[..., 0]*p[2] + d[..., 2]*p[9] + d[..., 3]*p[10]
            slow = d[..., 1]*p[3]
            jacobian = np.concatenate((fast[..., None], slow[..., None], a), axis=2)
            return (jacobian/scale[..., None]).reshape(-1, 11)

        return least_squares(fun, p, jac=jac, bounds=(low, high), loss=loss, x_scale='jac', max_nfev=120)

    best = None
    for pair in ((.35, 2.3), (.7, 3.8), (.18, 1.25)):
        logtau = np.log(pair)
        a, _ = _design(logtau, *inputs)
        w = np.linalg.lstsq(a.reshape(-1, 9), y.ravel(), rcond=None)[0]
        fit = optimize(np.r_[logtau, w], np.asarray(.09), 'soft_l1')
        if fit.x[0] < fit.x[1] and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise RuntimeError('No fit in fast-to-slow branch order')
    p = best.x

    def mad(v):
        return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))

    for _ in range(2):
        a, _ = _design(p[:2], *inputs)
        r = a @ p[2:] - y
        scale = 2.385*np.r_[np.full(len(times), mad(r[:, :-1])), mad(r[:, -1])]
        refined = optimize(p, scale, 'cauchy').x
        if refined[0] < refined[1]:
            p = refined
    return p


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x = _align(train, test)
    cal = _calibration(dataset['calibration_load_cells'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']))
    # Estimate material parameters from measured features, not cell-mean substitutes.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    inputs = _inputs(dataset, train, x[:n], cal)
    fit_inputs = tuple(v[complete] if i in (0, 1, 2, 5, 6) else v
                       for i, v in enumerate(inputs))
    p = _fit(fit_inputs, y[complete])
    a, _ = _design(p[:2], *_inputs(dataset, test, x[n:], cal))
    pred = a @ p[2:]
    return {'estimated_taus': np.exp(p[:2]), 'hereditary_weights': p[2:9],
            'regime_mode_weights': p[9:11], 'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
