"""Calibrated two-branch Prony model for piecewise-constant load histories."""

import numpy as np
from scipy.optimize import least_squares


def _load_bridge(cal):
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    physical = np.asarray(cal['true_channel_loads'], dtype=float)
    channel_map = np.linalg.lstsq(np.c_[raw, np.ones(len(raw))], physical, rcond=None)[0]
    return channel_map.sum(axis=1)


def _prepare(group, bridge, reference=None):
    lab = np.asarray(group['lab_id'])
    stratum = np.asarray(group['stratum'])
    x = np.array(group['coupon_features_raw'], dtype=float, copy=True)
    complete = np.isfinite(x).all(axis=1)
    reference = group if reference is None else reference
    panel = np.asarray(reference['ref_coupons'], dtype=float)
    panel_lab = np.asarray(reference['lab_id'])
    for key in np.unique(lab):
        rows = lab == key
        ref = panel[panel_lab == key]
        if len(ref) < 2 and reference is not group:
            ref = np.asarray(group['ref_coupons'])[rows]
        if len(ref) < 2:
            ref = panel
        center = np.median(ref.mean(axis=0))
        scale = np.median(ref.std(axis=0, ddof=0))
        x[rows] = (x[rows] - center) / (scale if scale > 0 else 1.)
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    weight = 1. / counts[cell.ravel(), None]
    observed = np.isfinite(x)
    filled = np.where(observed, x, 0.)
    mass = (weight * observed).sum(axis=0)
    mean = np.divide((weight * filled).sum(axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    drift = np.c_[np.ones(len(x)), np.where(observed, x, mean)]
    past = np.asarray(group['past_channel_raw']) @ bridge[:-1] + bridge[-1]
    hold = np.asarray(group['hold_channel_raw']) @ bridge[:-1] + bridge[-1]
    return drift, past, hold, complete


def _design(prepared, times, dt, taus):
    drift, past, hold, _ = prepared
    lag = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (np.exp(-lag[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    amplitude = hold[:, None] - memory
    creep = amplitude[:, None, :] * -np.expm1(-times[None, :, None] / taus)
    curve = np.concatenate((creep, times[None, :, None] * drift[:, None, :]), axis=2)
    rate = np.c_[amplitude / taus, drift]
    return curve, rate


def _fit(group, prepared, times, dt, pilot=False, mask=None):
    rows = np.ones(len(prepared[0]), dtype=bool) if mask is None else mask.copy()
    complete = rows & prepared[3]
    if complete.sum() >= 8 and np.linalg.matrix_rank(prepared[0][complete]) == 5:
        rows = complete
    data = tuple(a[rows] for a in prepared)
    y = np.asarray(group['strain_curves_obs'])[rows] - np.asarray(group['strain_0'])[rows, None]
    r = np.asarray(group['instant_rate_obs'])[rows]
    rate_weight = .7
    target = np.r_[y.ravel(), rate_weight * r]

    def matrix(log_taus):
        curve, rate = _design(data, times, dt, np.exp(log_taus))
        return np.r_[curve.reshape(-1, 7), rate_weight * rate]

    def residual(par):
        return matrix(par[:2]) @ par[2:] - target

    lower = [np.log(.050001)] * 2 + [-np.inf] * 7
    upper = [np.log(24.9999)] * 2 + [np.inf] * 7
    best = None
    for start in ((.32, 2.), (.75, 5.), (1.4, 12.)):
        z = np.log(start)
        linear = np.linalg.lstsq(matrix(z), target, rcond=None)[0]
        fit = least_squares(residual, np.r_[z, linear], bounds=(lower, upper),
                            loss='soft_l1', f_scale=.004 if pilot else .012,
                            max_nfev=140, x_scale='jac')
        if best is None or fit.cost < best.cost:
            best = fit
    order = np.argsort(best.x[:2])
    taus = np.exp(best.x[:2][order])
    if taus[0] == taus[1]:
        taus[0] = np.nextafter(taus[0], 0.)
    weights = np.r_[best.x[2:4][order], best.x[4:]]
    return taus, weights


def solve_batch(dataset: dict) -> dict:
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    bridge = _load_bridge(dataset['calibration_load_cells'])
    pilot = dataset.get('chamber_verification_trials')
    use_pilot = pilot is not None and len(pilot['instant_rate_obs']) >= 8
    training = pilot if use_pilot else dataset['train']
    taus, weights = _fit(training, _prepare(training, bridge), times, dt, pilot=use_pilot)
    test = dataset['test']
    a, b = _design(_prepare(test, bridge), times, dt, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
