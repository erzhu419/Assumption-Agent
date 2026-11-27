"""Fit intrinsic relaxation from calibrated benchtop coupons."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x, refs, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for label in np.unique(lab):
        mask = lab == label
        panel = refs[mask]
        x[mask] = (x[mask] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1.0 / counts[cell]
    for col in x.T:
        known = np.isfinite(col)
        col[~known] = np.average(col[known], weights=w[known])
    return np.split(x, [len(train['lab_id'])])


def _loads(split):
    return tuple(split[k].sum(axis=-1) for k in ('past_channel_raw', 'hold_channel_raw'))


def _gap(taus, past, hold, dt):
    history = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        history += -np.expm1(-width / taus) * (stress[:, None] - history)
    return hold[:, None] - history


def _basis(taus, past, hold, x, dt, t):
    gap = _gap(taus, past, hold, dt)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            t[None, :, None] * drift[:, None, :]), axis=-1)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _material(train, x, dt, t):
    past, hold = _loads(train)
    y = (train['strain_curves_obs'] - train['strain_0'][:, None]).ravel()
    r = train['instant_rate_obs']
    target = np.r_[y, r]

    def design(taus):
        curve, rate = _basis(taus, past, hold, x, dt, t)
        return np.vstack((curve.reshape(-1, 7), rate))

    def residual(p):
        return np.einsum('ij,j->i', design(np.exp(p[:2])), p[2:]) - target

    bounds = (np.r_[np.log([0.050001] * 2), [-np.inf] * 7],
              np.r_[np.log([24.99999] * 2), [np.inf] * 7])
    candidates = []
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


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt, t = cal_item['past_dt'], cal_item['eval_times']
    x_train, x_test = _features(train, test)
    taus, theta = _material(train, x_train, dt, t)
    curve, rate = _basis(taus, *_loads(test), x_test, dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
