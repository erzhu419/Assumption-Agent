"""Fit the intrinsic material law and apply the characterized chamber boundary."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x, refs, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for label in np.unique(lab):
        mask = lab == label
        panel = refs[mask]
        center = np.median(np.mean(panel, axis=0))
        scale = np.median(np.std(panel, axis=0))
        x[mask] = (x[mask] - center) / scale
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cell]
    for col in x.T:
        known = np.isfinite(col)
        col[~known] = np.average(col[known], weights=weights[known])
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _loads(split, bridge):
    return tuple(split[k] @ bridge[1:] + bridge[0] for k in
                 ('past_channel_raw', 'hold_channel_raw'))


def _basis(taus, past, hold, x, dt, t):
    history = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        gain = -np.expm1(-width / taus)
        history += gain * (stress[:, None] - history)
    gap = hold[:, None] - history
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            t[None, :, None] * drift[:, None, :]), axis=-1)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _material(train, bridge, x, dt, t):
    past, hold = _loads(train, bridge)
    y = (train['strain_curves_obs'] - train['strain_0'][:, None]).ravel()
    r = train['instant_rate_obs']
    target = np.r_[y, r]
    def design(taus):
        curve, rate = _basis(taus, past, hold, x, dt, t)
        return np.vstack((curve.reshape(-1, 7), rate))
    def residual(p):
        return design(np.exp(p[:2])) @ p[2:] - target
    candidates = []
    bounds = (np.r_[np.log([0.050001] * 2), [-np.inf] * 7],
              np.r_[np.log([24.99999] * 2), [np.inf] * 7])
    for seed in ([0.35, 2.0], [0.8, 5.0], [0.2, 10.0]):
        theta = np.linalg.lstsq(design(np.array(seed)), target, rcond=None)[0]
        candidates.append(least_squares(residual, np.r_[np.log(seed), theta],
                                        bounds=bounds, loss='soft_l1', f_scale=0.015))
    fit = min(candidates, key=lambda q: q.cost)
    err = residual(fit.x)
    def scale(a):
        return max(0.0005, 1.4826 * np.median(np.abs(a - np.median(a))))
    noise = np.r_[np.full(len(y), scale(err[:len(y)])), np.full(len(r), scale(err[len(y):]))]
    fit = least_squares(lambda p: residual(p) / noise, fit.x, bounds=bounds,
                        loss='soft_l1', f_scale=1.5)
    taus, theta = np.exp(fit.x[:2]), fit.x[2:].copy()
    order = np.argsort(taus)
    theta[:2] = theta[:2][order]
    return taus[order], theta


def _chamber_features(coupons):
    x = np.array(coupons['coupon_features_raw'], dtype=float, copy=True)
    lab = np.asarray(coupons['lab_id'])
    refs = np.asarray(coupons['ref_coupons'])
    for label in np.unique(lab):
        mask = lab == label
        panel = refs[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0))
        x[mask] = (x[mask] - center) / scale
    return np.where(np.isnan(x), np.nanmean(x, axis=0), x)


def _predict_chamber(coupons, taus, theta, dt, t):
    # The bridge has already been inverted upstream; sum its physical channels.
    past, hold = _loads(coupons, np.r_[0., np.ones(4)])
    x = _chamber_features(coupons)
    curve_basis, rate_basis = _basis(taus, past, hold, x, dt, t)
    drift = theta[2] + x @ theta[3:]
    # Chamber drift relaxes exponentially with unit time constant and unit
    # initial slope. The hereditary stress branches retain their full history.
    strain = (coupons['strain_0'][:, None] + curve_basis[:, :, :2] @ theta[:2]
              - drift[:, None] * np.expm1(-t))
    rate = rate_basis[:, :2] @ theta[:2] + drift
    return strain, rate


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt = np.asarray(cal_item['past_dt'], dtype=float)
    t = np.asarray(cal_item['eval_times'], dtype=float)
    x_train, _ = _features(train, test)
    taus, theta = _material(train, np.r_[0., np.ones(4)], x_train, dt, t)
    strain, rate = _predict_chamber(test, taus, theta, dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
