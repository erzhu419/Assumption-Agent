"""Calibrated two-branch rheology with saturating jammed slip."""

import json
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares


_HISTORY = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _history_rows(node):
    if isinstance(node, dict):
        keys = ('lab_id', 'cmd_stress', 'true_stress')
        if all(k in node for k in keys):
            yield from zip(*(np.atleast_1d(node[k]) for k in keys))
        else:
            for value in node.values():
                yield from _history_rows(value)
    elif isinstance(node, list):
        for value in node:
            yield from _history_rows(value)


def _calibration(data):
    c = data['calibration_load_cells']
    lab_ids = np.asarray(c['lab_id'])
    cmd, true = np.asarray(c['cmd_stress']), np.asarray(c['true_stress'])
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        keep = (lab_ids == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[keep], true[keep]
        spike = False
        sparse = len(x) in (3, 4)
        if sparse:
            # Screen pair extrapolations with a margin for calibration noise.
            # A recent archived slope confirms suspected EMI below.
            for j, k in combinations(range(len(x)), 2):
                dx = x[k] - x[j]
                if dx != 0:
                    residual = y - y[j] - (y[k] - y[j]) / dx * (x - x[j])
                    if np.max(np.abs(residual)) > 2.90:
                        spike = True
                        break
        elif len(x) >= 5:
            dx, dy = x[:, None] - x, y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full(dx.shape, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            offset = y - gain*x
            clean = np.abs(offset - np.median(offset)) <= 3.0
            spike = not clean.all()
            x, y = x[clean], y[clean]
        if len(x) >= 2 and (not spike or (not sparse and len(x) >= 4)):
            result[lab] = _line(x, y)
        else:
            if history is None:
                with open(_HISTORY) as file:
                    history = json.load(file)
            # A hardware change separates the identity sweep from the recent prefix.
            # Use the latest sweep that can estimate this channel's gain.
            for sweep in reversed(history):
                pairs = np.array([(u, v) for c, u, v in _history_rows(sweep)
                                  if int(c) == lab and np.isfinite(u) and np.isfinite(v)])
                if len(pairs) >= 2 and np.ptp(pairs[:, 0]) > 0:
                    gain, bias = _line(*pairs.T)
                    break
            else:
                raise ValueError(f'No usable archived calibration for channel {lab}')
            if spike and sparse:
                offsets = y - gain*x
                clean = np.abs(offsets - np.median(offsets)) <= 3.0
                if clean.all():
                    result[lab] = _line(x, y)
                    continue
                x, y = x[clean], y[clean]
            if len(x):
                bias = np.mean(y - gain*x)
            result[lab] = gain, bias
    return result


def _align(data):
    train, test = data['train'], data['test']
    x, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for lab in np.unique(labs):
        take = labs == lab
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - center)/scale
    cells = []
    for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
        part = x[(labs == lab) & (strata == stratum)]
        counts = np.isfinite(part).sum(axis=0)
        cells.append(np.divide(np.nansum(part, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(x), x, np.nanmean(cells, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (split['past_stress_cmd']*gain[:, None] + bias[:, None],
            split['stress_hold_cmd']*gain + bias)


def _design(tau, past, hold, dt, time, features, jammed):
    decay = np.exp(-dt[:, None]/tau)
    update = -np.expm1(-dt[:, None]/tau)
    memory = np.zeros((len(hold), 2))
    for cmd, a, b in zip(past.T, decay, update):
        memory = a*memory + b*cmd[:, None]
    g0, g1 = (hold[:, None] - memory).T
    a0, a1 = (-np.expm1(-time[:, None]/tau)).T
    phi = time/(1.0 + 0.85*time) if jammed else time
    curves = np.zeros((len(hold), len(time), 9))
    rates = np.zeros((len(hold), 9))
    curves[:, :, 0], curves[:, :, 1] = g0[:, None]*a0, g1[:, None]*a1
    rates[:, 0], rates[:, 1] = g0/tau[0], g1/tau[1]
    curves[:, :, 2:7] = phi[None, :, None]*features[:, None, :]
    rates[:, 2:7] = features
    if jammed:
        slip = np.sign(hold)*np.sqrt(np.abs(hold))
        change = g0 - g1
        curves[:, :, 7], curves[:, :, 8] = slip[:, None]*phi, change[:, None]*a0
        rates[:, 7], rates[:, 8] = slip, change/tau[0]
    return curves, rates


def _taus(p):
    return np.exp([p[0], p[0] + p[1]])


def _mad(residual):
    return max(1e-8, 1.4826*np.median(np.abs(residual - np.median(residual))))


def _fit(past, hold, dt, time, features, y_curve, y_rate, jammed):
    n_curve = y_curve.size
    n_weights = 9 if jammed else 7
    y = np.r_[y_curve.ravel(), y_rate]
    good = np.isfinite(y)
    y = y[good]
    is_curve = (np.arange(len(good)) < n_curve)[good]

    def matrix(p):
        curves, rates = _design(_taus(p), past, hold, dt, time, features, jammed)
        return np.concatenate((curves.reshape(-1, 9), rates))[good, :n_weights]

    def initialize(pair, weights):
        p = np.log([pair[0], pair[1]/pair[0]])
        a = matrix(p)
        return np.r_[p, np.linalg.lstsq(a*weights[:, None], y*weights, rcond=None)[0]]

    def unscaled(p):
        return matrix(p) @ p[2:] - y

    def thresholds(residual):
        return np.where(is_curve, 2.3*_mad(residual[is_curve]), 2.3*_mad(residual[~is_curve]))

    bounds = (np.r_[np.log(0.015), np.log(1.003), np.full(n_weights, -np.inf)],
              np.r_[np.log(30.), np.log(300.), np.full(n_weights, np.inf)])
    starts = [(0.35, 1.8), (0.18, 0.8), (0.8, 4.0), (1.5, 9.0)]
    pilot = initialize(starts[0], np.ones_like(y))
    scale = thresholds(unscaled(pilot))

    def fit(p, loss):
        return least_squares(lambda z: unscaled(z)/scale, p, bounds=bounds, loss=loss,
                             x_scale='jac', max_nfev=100)

    pilot = fit(pilot, 'soft_l1').x
    scale = thresholds(unscaled(pilot))
    best = fit(pilot, 'cauchy')
    for pair in starts[1:]:
        weights = 1/(scale*np.sqrt(1 + (unscaled(best.x)/scale)**2))
        other = fit(initialize(pair, weights), 'cauchy')
        if other.cost < best.cost:
            best = other
    scale = thresholds(unscaled(best.x))
    p = fit(best.x, 'cauchy').x
    weights = np.zeros(9)
    weights[:n_weights] = p[2:]
    return _taus(p), weights


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    calibration = _calibration(dataset)
    x = _align(dataset)
    features = np.column_stack((np.ones(len(x)), x))
    n_train = len(train['lab_id'])
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    jammed = dataset['shear_banding_index'] > 0.50
    past, hold = _physical(train, calibration)
    taus, weights = _fit(past, hold, dt, time, features[:n_train],
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'], jammed)
    past, hold = _physical(test, calibration)
    curves, rates = _design(taus, past, hold, dt, time, features[n_train:], jammed)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
