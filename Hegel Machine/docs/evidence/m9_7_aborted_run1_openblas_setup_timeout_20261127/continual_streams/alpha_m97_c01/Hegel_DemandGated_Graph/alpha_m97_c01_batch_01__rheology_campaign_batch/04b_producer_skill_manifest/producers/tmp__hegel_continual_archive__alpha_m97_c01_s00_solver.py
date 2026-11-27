"""Calibrate and fit the two-branch linear-flow rheology model."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(lab):
    if not _ARCHIVE.is_file():
        return None
    history = json.loads(_ARCHIVE.read_text())
    if isinstance(history, dict):
        history = history.get('calibration_load_cells', history)
        if 'lab_id' in history:
            history = [dict(zip(history, row)) for row in zip(*history.values())]
        else:
            record = history.get(str(lab))
            history = [record] if isinstance(record, dict) else (record or [])
    rows = [r for r in history if int(r.get('lab_id', lab)) == lab]
    affine = [(r['gain'], r['bias']) for r in rows if 'gain' in r and 'bias' in r]
    if affine:
        return np.median(affine, axis=0)
    pairs = [(r['cmd_stress'], r['true_stress']) for r in rows if 'cmd_stress' in r and 'true_stress' in r]
    if len(pairs) >= 2:
        x, y = np.asarray(pairs).T
        if np.ptp(x) > 0:
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


def _design(logtau, data, dt, time):
    past, hold, x = data
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for width, stress in zip(dt, past.T):
        u = width/tau
        a = np.exp(-u)
        dz = a*dz + (u*a)*(z-stress[:, None])
        z = a*z - np.expm1(-u)*stress[:, None]
    gap = hold[:, None] - z
    u = time[:, None]/tau
    f = -np.expm1(-u)
    a = np.empty((len(hold), len(time), 7))
    a[:, :, :2] = gap[:, None, :]*f
    a[:, :, 2] = time
    a[:, :, 3:] = time[None, :, None]*x[:, None, :]
    jac = -dz[:, None, :]*f - gap[:, None, :]*u*np.exp(-u)
    return a, jac, gap


def _fit(data, dt, time, delta, rate_obs=None):
    good = np.isfinite(delta)
    y = delta[good]
    # Coarse profile over ordered times, then refine the best separated seeds.
    lo = np.log(max(np.min(dt)*0.12, 1e-4))
    hi = np.log(max(np.max(time), np.sum(dt))*12.)
    grid = np.geomspace(max(np.min(dt)*0.7, 1e-4), max(np.max(time), np.sum(dt))*1.25, 8)
    seeds = []
    scale_y = max(np.median(np.abs(y - np.median(y)))*1.4826, 1e-6)
    for i, short in enumerate(grid[:-1]):
        for long in grid[i+1:]:
            logs = np.log([short, long])
            a = _design(logs, data, dt, time)[0][good]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for factor in (0.20, 0.04):
                resid = y-a@b
                w = 1./np.hypot(1., resid/(scale_y*factor))
                b = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
            score = np.median(np.abs(y-a@b))
            seeds.append((score, np.r_[logs, b]))
    seeds.sort(key=lambda item: item[0])
    best, runner = seeds[0], next((s for s in seeds[1:] if np.linalg.norm(s[1][:2]-seeds[0][1][:2]) > 0.9), seeds[1])
    def residual(p):
        return _design(p[:2], data, dt, time)[0][good]@p[2:] - y
    def jacobian(p):
        a, jac, _ = _design(p[:2], data, dt, time)
        return np.column_stack((jac[good]*p[2:4], a[good]))
    bounds = (np.r_[lo, lo, np.full(7, -np.inf)], np.r_[hi, hi, np.full(7, np.inf)])
    scale = max(1.2*best[0], 1e-7*scale_y)
    solutions = []
    for _, initial in (best, runner):
        p = initial
        for factor in (4., 1.):
            opt = least_squares(residual, p, jac=jacobian, bounds=bounds, x_scale='jac', loss='cauchy', f_scale=scale*factor, max_nfev=120)
            p = opt.x
        solutions.append((opt.cost, p))
    params = min(solutions, key=lambda item: item[0])[1]
    if rate_obs is not None:
        rate_good = np.isfinite(rate_obs)
        if np.any(rate_good):
            _, rate = _predict(params[:2], params[2:], data, dt, time)

            def noise_scale(error):
                return max(1e-7, 1.4826*np.median(np.abs(error - np.median(error))))

            curve_scale = noise_scale(residual(params))
            rate_scale = noise_scale((rate - rate_obs)[rate_good])

            def joint_residual(p):
                curve, rate = _predict(p[:2], p[2:], data, dt, time)
                return np.r_[((curve - delta)[good])/curve_scale,
                             ((rate - rate_obs)[rate_good])/rate_scale]

            params = least_squares(joint_residual, params, bounds=bounds,
                                   x_scale='jac', loss='cauchy', max_nfev=100).x
    order = np.argsort(params[:2])
    return params[:2][order], np.r_[params[2:4][order], params[4:]]


def _predict(logtau, hw, data, dt, time):
    a, _, gap = _design(logtau, data, dt, time)
    rate = gap@(hw[:2]/np.exp(logtau)) + hw[2] + data[2]@hw[3:7]
    return a@hw, rate


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    dt, times = dataset['past_dt'], dataset['eval_times']
    data, n = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, weights = _fit(tuple(v[:n] for v in data), dt, times, delta,
                         train.get('instant_rate_obs'))
    strain, rate = _predict(logs, weights, tuple(v[n:] for v in data), dt, times)
    return {'estimated_taus': np.exp(logs), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
