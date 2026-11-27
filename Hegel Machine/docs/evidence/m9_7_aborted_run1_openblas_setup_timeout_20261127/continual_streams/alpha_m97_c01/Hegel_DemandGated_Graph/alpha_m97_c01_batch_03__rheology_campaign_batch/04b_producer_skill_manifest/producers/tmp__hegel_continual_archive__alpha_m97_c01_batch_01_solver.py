"""Fit calibrated two-branch creep and predict strain and onset rate."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    dx = x[:, None] - x
    dy = y[:, None] - y
    # The repeated-median line screens EMI without allowing spikes to pull OLS.
    center = np.median([np.median(v[u != 0] / u[u != 0])
                        for u, v in zip(dx, dy)])
    bias = np.median(y - center*x)
    return np.abs(y - center*x - bias) <= 3.0


def _calibrate(c):
    lab, cmd, true = (np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.zeros((4, 2))
    history = None
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            clean = _line(x, y)
            spikes = not clean.all()
            x, y = x[clean], y[clean]
        if len(x) >= 4 and np.ptp(x) > 0:
            result[channel] = np.polyfit(x, y, 1)
            continue
        if len(x) < 2 or spikes:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            # Use the latest usable prefix sweep if the batch cannot resolve a line.
            for sweep in reversed(history):
                take = np.asarray(sweep['lab_id']) == channel
                p = np.asarray(sweep['cmd_stress'])[take]
                q = np.asarray(sweep['true_stress'])[take]
                if len(p) >= 3 and np.ptp(p) > 0:
                    clean = _line(p, q)
                    p, q = p[clean], q[clean]
                if len(p) >= 2 and np.ptp(p) > 0:
                    result[channel] = np.polyfit(p, q, 1)
                    break
            else:
                raise ValueError(f'No usable calibration for load cell {channel}')
        else:
            result[channel] = np.polyfit(x, y, 1)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, strata = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x = join('coupon_features_raw').astype(float)
    ref = join('ref_coupons')
    for channel in np.unique(lab):
        take = lab == channel
        panel = ref[take]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - loc) / scale
    means = np.array([np.nanmean(x[(lab == i) & (strata == j)], axis=0)
                      for i, j in np.unique(np.column_stack((lab, strata)), axis=0)])
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    act = np.tanh(np.maximum(np.abs(past) - dataset['crit_stress'], 0)**2 @ dataset['past_dt'])
    if dataset['shear_banding_index'] > .5:
        act[:] = 0.
    return past, hold, x, act


def _design(logtau, data, dt, time, mode):
    past, hold, x, act = data
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for width, stress in zip(dt, past.T):
        u = width/tau
        a = np.exp(-u)
        dz = a*dz + u*a*(z - stress[:, None])
        z = a*z - np.expm1(-u)*stress[:, None]
    gap = hold[:, None] - z
    u = time[:, None]/tau
    f = -np.expm1(-u)
    df = -u*np.exp(-u)
    plateau = np.log1p(time)
    terminal = time if mode == 'linear' else plateau
    a = np.empty((len(hold), len(time), 9))
    a[:, :, :2] = gap[:, None, :]*f
    a[:, :, 2] = terminal
    a[:, :, 3:7] = a[:, :, 2, None]*x[:, None, :]
    a[:, :, 7] = (act*hold)[:, None]*f[:, 0]
    a[:, :, 8] = (act*z[:, 0])[:, None]*plateau
    j = -dz[:, None, :]*f + gap[:, None, :]*df
    j0 = (act*hold)[:, None]*df[:, 0]
    j1 = (act*dz[:, 0])[:, None]*plateau
    r = np.column_stack((gap/tau, np.ones(len(hold)), x, act*hold/tau[0], act*z[:, 0]))
    jr = (-dz - gap)/tau
    return a, r, j, j0, j1, jr, act*dz[:, 0]


def _scale(v):
    return max(1.4826*np.median(np.abs(v - np.median(v))), 1e-7)


def _dot(a, b):
    return np.einsum('...j,j->...', a, b)


def _fit(data, dt, time, delta, rate, mode):
    ys, yr = np.isfinite(delta), np.isfinite(rate)
    y = delta[ys]
    lower = np.log(max(np.min(dt)*.1, 1e-4))
    upper = np.log(max(np.max(time), np.sum(dt))*12)
    grid = np.geomspace(max(np.min(dt)*.55, 1e-4), max(np.max(time), np.sum(dt))*1.25, 9)
    seeds = []
    scale_y = _scale(y)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i+1:]:
            logs = np.log([fast, slow])
            a = _design(logs, data, dt, time, mode)[0][ys]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for width in (.2, .06, .02):
                rootw = 1./np.sqrt(np.hypot(1., (_dot(a, b)-y)/(width*scale_y)))
                b = np.linalg.lstsq(a*rootw[:, None], y*rootw, rcond=None)[0]
            seeds.append((np.median(np.abs(_dot(a,b) - y)), np.r_[logs, b]))
    seeds.sort(key=lambda s: s[0])
    initial = seeds[0][1]
    runner = next(s[1] for s in seeds[1:] if np.linalg.norm(s[1][:2]-initial[:2]) > .8)
    bnd = (np.r_[lower, lower, np.full(9, -np.inf)], np.r_[upper, upper, np.full(9, np.inf)])

    def components(p):
        return _design(p[:2], data, dt, time, mode)

    def residual(p, sc, sr):
        a, r, *_ = components(p)
        return np.r_[(_dot(a[ys], p[2:]) - y)/sc, (_dot(r[yr], p[2:]) - rate[yr])/sr]

    def jacobian(p, sc, sr):
        a, r, j, j0, j1, jr, jr1 = components(p)
        jc = j*p[2:4]
        jc[:, :, 0] += j0*p[9] + j1*p[10]
        jrate = jr*p[2:4]
        jrate[:, 0] += -r[:, 7]*p[9] + jr1*p[10]
        return np.vstack((np.column_stack((jc[ys], a[ys]))/sc,
                          np.column_stack((jrate[yr], r[yr]))/sr))

    a, r, *_ = components(initial)
    sc = _scale(_dot(a[ys], initial[2:]) - y)
    sr = _scale(_dot(r[yr], initial[2:]) - rate[yr]) if yr.any() else 1.
    opts = []
    for seed in (initial, runner):
        for width in (3., 1.5):
            opt = least_squares(residual, seed, jac=jacobian, args=(sc, sr), bounds=bnd,
                                loss='cauchy', f_scale=width, x_scale='jac', max_nfev=120)
            seed = opt.x
        opts.append(opt)
    best = min(opts, key=lambda opt: opt.cost).x
    a, r, *_ = components(best)
    sc = _scale(_dot(a[ys], best[2:]) - y)
    sr = _scale(_dot(r[yr], best[2:]) - rate[yr]) if yr.any() else 1.
    return least_squares(residual, best, jac=jacobian, args=(sc, sr), bounds=bnd,
                         loss='cauchy', f_scale=2., x_scale='jac', max_nfev=100).x


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    data = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    mode = 'log' if dataset['shear_banding_index'] <= .5 else 'linear'
    p = _fit(tuple(a[:n] for a in data), dt, time, delta, train['instant_rate_obs'], mode)
    a, r, *_ = _design(p[:2], tuple(a[n:] for a in data), dt, time, mode)
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:9], regime_mode_weights=p[9:],
                predicted_strain=test['strain_0'][:, None] + _dot(a, p[2:]),
                initial_rate_pred=_dot(r, p[2:]))
