"""Fit the calibrated two-branch crystallization response."""

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data):
    c = data['calibration_load_cells']
    result = np.empty((4, 2))
    for lab in range(4):
        take = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][take], c['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            clean = np.abs(y - gain*x - np.median(y - gain*x)) <= 3.0
        if clean.sum() >= 2 and (clean.all() or clean.sum() >= 4):
            result[lab] = _line(x[clean], y[clean])
        else:
            result[lab] = _archive_calibration(lab, x[clean], y[clean])
    return result


def _archive_calibration(lab, x, y):
    import json
    try:
        with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as file:
            records = json.load(file)
    except FileNotFoundError:
        return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)

    def rows(node):
        if isinstance(node, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= node.keys():
                yield from zip(*(np.atleast_1d(node[key]) for key in ('lab_id', 'cmd_stress', 'true_stress')))
            else:
                for value in node.values():
                    yield from rows(value)
        elif isinstance(node, list):
            for value in node:
                yield from rows(value)

    pairs = np.array([(u, v) for c, u, v in rows(records) if int(c) == lab and np.isfinite(u) and np.isfinite(v)]).reshape(-1, 2)
    if len(pairs) >= 2:
        gain, bias = _line(*pairs.T)
        if len(x):
            bias = np.mean(y - gain*x)
        return gain, bias
    return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)


def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    n_train = len(train['lab_id'])
    return aligned[:n_train], aligned[n_train:]


def _prepare(data):
    calibration = _calibration(data)
    features = _align(data)
    dt = np.asarray(data['past_dt'])
    t = np.asarray(data['eval_times'])
    plateau = data['shear_banding_index'] <= 0.50
    blocks = []
    for split, aligned in zip(('train', 'test'), features):
        s = data[split]
        gain, bias = calibration[s['lab_id']].T
        past = s['past_stress_cmd'] * gain[:, None] + bias[:, None]
        hold = s['stress_hold_cmd'] * gain + bias
        damage = (np.maximum(np.abs(past) - data['crit_stress'], 0.0)**2) @ dt
        act = np.tanh(damage) if plateau else np.zeros_like(damage)
        blocks.append((past, hold, np.column_stack((np.ones(len(hold)), aligned)), act, np.full(len(hold), plateau)))
    return dt, t, blocks


def _basis(taus, dt, t, block):
    past, hold, coupons, act, plateau = block
    dtau = dt[:, None] / taus
    decay = np.exp(-dtau)
    inject = -np.expm1(-dtau)
    memory = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        memory = decay[s] * memory + inject[s] * past[:, s, None]
    gap = hold[:, None] - memory
    rise = -np.expm1(-t[:, None] / taus)
    # Low shear selects the terminal time law; work damage scales only the two corrections.
    terminal = np.where(plateau[:, None], np.log1p(t), t)
    crystallized_hold = act * hold
    crystallized_memory = act * memory[:, 0]
    curves = np.concatenate((gap[:, None] * rise,
                             coupons[:, None] * terminal[:, :, None],
                             (crystallized_hold[:, None] * rise[:, 0])[:, :, None],
                             (crystallized_memory[:, None] * terminal)[:, :, None]), axis=2)
    rates = np.column_stack((gap / taus, coupons, crystallized_hold / taus[0], crystallized_memory))
    return curves, rates


def _fit(dt, t, block, y, rate):
    target = np.concatenate((y.ravel(), rate))
    base = np.concatenate((np.ones(y.size), np.full(rate.size, 2.0)))
    # Isolated strain spikes should not outweigh the rest of the same curve.
    scale = np.concatenate((np.tile(0.025 * (1.0 + 0.35 * t), len(y)),
                            np.full(rate.size, 0.05)))
    sqrt_weight = base

    def project(log_taus, return_weights=False):
        curves, rates = _basis(np.exp(log_taus), dt, t, block)
        design = np.concatenate((curves.reshape(-1, 9), rates))
        weights = np.linalg.lstsq(design * sqrt_weight[:, None], target * sqrt_weight,
                                  rcond=None)[0]
        residual = np.einsum('ij,j->i', design, weights) - target
        return (weights, residual) if return_weights else residual * sqrt_weight

    bounds = (np.log(0.005), np.log(250.0))
    starts = np.log([[0.45, 2.7], [0.8, 7.0]])
    for _ in range(12):
        fit = min((least_squares(project, start, bounds=bounds, max_nfev=45)
                   for start in starts), key=lambda result: result.cost)
        weights, residual = project(fit.x, return_weights=True)
        updated = base / np.hypot(1.0, residual / scale)
        if np.max(np.abs(updated - sqrt_weight)) < 1e-4:
            break
        sqrt_weight = updated
        starts = [fit.x]
    return np.exp(fit.x), weights


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    dt, t, (train, test) = _prepare(dataset)
    s = dataset['train']
    y = s['strain_curves_obs'] - s['strain_0'][:, None]
    taus, weights = _fit(dt, t, train, y, s['instant_rate_obs'])
    curves, rates = _basis(taus, dt, t, test)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': dataset['test']['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
