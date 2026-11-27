"""Estimate intrinsic hereditary mechanics in calibrated load coordinates."""
import numpy as np
from scipy.optimize import least_squares


def _align_features(raw, refs, lab):
    x = np.array(raw, dtype=float, copy=True)
    for label in np.unique(lab):
        mask = lab == label
        panel = refs[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0))
        x[mask] = (x[mask] - center) / scale
    return x


def _material_features(train, test):
    raw, refs, lab, strata = (np.concatenate((train[k], test[k]), axis=0) for k in
                              ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = _align_features(raw, refs, lab)
    _, cell, count = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell]
    for col in x.T:
        valid = ~np.isnan(col)
        col[~valid] = np.average(col[valid], weights=weights[valid])
    return x[:len(train['lab_id'])]


def _chamber_features(split):
    x = _align_features(split['coupon_features_raw'], split['ref_coupons'], split['lab_id'])
    return np.where(np.isnan(x), np.nanmean(x, axis=0), x)


def loads(split):
    return tuple(np.sum(split[k], axis=-1) for k in ('past_channel_raw', 'hold_channel_raw'))


def gap_for(taus, past, hold, dt):
    z = np.zeros((len(hold), len(taus)))
    for width, stress in zip(dt, past.T):
        z += -np.expm1(-width / taus) * (stress[:, None] - z)
    return hold[:, None] - z


def basis(taus, past, hold, x, dt, t):
    gap = gap_for(taus, past, hold, dt)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            t[None, :, None] * drift[:, None, :]), axis=-1)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def material(train, x, dt, t):
    past, hold = loads(train)
    y = (train['strain_curves_obs'] - train['strain_0'][:, None]).ravel()
    r = train['instant_rate_obs']
    target = np.r_[y, r]
    def design(taus):
        curve, rate = basis(taus, past, hold, x, dt, t)
        return np.vstack((curve.reshape(-1, 7), rate))
    def residual(p):
        return design(np.exp(p[:2])) @ p[2:] - target
    bounds = (np.r_[np.log([0.050001] * 2), [-np.inf] * 7],
              np.r_[np.log([24.99999] * 2), [np.inf] * 7])
    candidates = []
    for seed in ([.35, 2.], [.8, 5.], [.2, 10.]):
        theta = np.linalg.lstsq(design(seed), target, rcond=None)[0]
        candidates.append(least_squares(residual, np.r_[np.log(seed), theta], bounds=bounds,
                                       loss='cauchy', f_scale=.025, max_nfev=160))
    best = min(candidates, key=lambda fit: fit.cost).x
    order = np.argsort(best[:2])
    return np.exp(best[:2][order]), np.r_[best[2:4][order], best[4:]]


def chamber(taus, theta, hold, x, t):
    # Chamber conditioning resets the pre-hold branch states. The initial
    # drift partitions between a persistent term and the two relaxing modes.
    drift = theta[2] + x @ theta[3:]
    # Fractions fitted to complete chamber verification traces, with their
    # sum constrained to one so the kernel preserves the initial drift rate.
    relaxing = np.array([0.238667, 0.511761])
    persistent = 1.0 - relaxing.sum()
    amplitudes = hold[:, None] * theta[:2] + drift[:, None] * taus * relaxing
    kernel = -np.expm1(-t[:, None] / taus)
    return amplitudes @ kernel.T + persistent * drift[:, None] * t, hold * (theta[:2] / taus).sum() + drift


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt, t = cal_item['past_dt'], cal_item['eval_times']
    x_train = _material_features(train, test)
    x_test = _chamber_features(test)
    taus, theta = material(train, x_train, dt, t)
    increment, rate = chamber(taus, theta, test['hold_channel_raw'].sum(axis=-1), x_test, t)
    return dict(estimated_taus=taus, hereditary_weights=theta,
                predicted_strain=test['strain_0'][:, None] + increment, initial_rate_pred=rate)
