"""Fit the benchtop relaxation spectrum and apply the chamber boundary law."""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq


def _features(*parts):
    lab, stratum, refs, raw = (np.concatenate([p[k] for p in parts]) for k in (
        'lab_id', 'stratum', 'ref_coupons', 'coupon_features_raw'))
    x = raw.astype(float, copy=True)
    for name in np.unique(lab):
        rows = lab == name
        panel = refs[rows]
        x[rows] = (x[rows] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = 1.0 / count[cell]
    present = np.isfinite(x)
    mass = weight @ present
    fill = np.divide(weight @ np.where(present, x, 0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    x = np.where(present, x, fill)
    cuts = np.cumsum([len(p['lab_id']) for p in parts[:-1]])
    return np.split(np.column_stack((np.ones(len(x)), x)), cuts)


def _basis(tau, times):
    return np.column_stack((-np.expm1(-times[:, None] / tau), times)), np.r_[1 / tau, 1.0]


def _design(tau, history, hold, x, dt, times):
    state = np.zeros((len(hold), 2))
    for duration, stress in zip(dt, history.T):
        uptake = -np.expm1(-duration / tau)
        state += uptake * (stress[:, None] - state)
    gap = hold[:, None] - state
    b, r = _basis(tau, times)
    curves = np.concatenate((gap[:, None, :] * b[None, :, :2],
                             times[None, :, None] * x[:, None, :]), axis=-1)
    return np.vstack((curves.reshape(-1, 7), np.column_stack((gap * r[:2], x))))


def _material(train, x, dt, times):
    h = np.asarray(train['hold_channel_raw']).sum(axis=-1)
    p = np.asarray(train['past_channel_raw']).sum(axis=-1)
    y = np.r_[(train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(),
              train['instant_rate_obs']]
    lower = np.log([.050001, .050001])
    upper = np.log([24.9999, 24.9999])

    def design(u):
        return _design(np.exp(u), p, h, x, dt, times)

    def projected(u):
        a = design(u)
        return a @ lstsq(a, y, lapack_driver='gelsy')[0] - y

    def residual(v):
        return design(v[:2]) @ v[2:] - y

    fit = min((least_squares(projected, np.log(start), bounds=(lower, upper),
                             max_nfev=40) for start in ((.25, 2.), (.65, 4.), (1.2, 9.))),
              key=lambda result: result.cost)
    w = lstsq(design(fit.x), y, lapack_driver='gelsy')[0]
    bounds = (np.r_[lower, np.full(7, -np.inf)], np.r_[upper, np.full(7, np.inf)])
    v = least_squares(residual, np.r_[fit.x, w], bounds=bounds,
                      loss='soft_l1', f_scale=.02, max_nfev=75).x
    v = least_squares(residual, v, bounds=bounds,
                      loss='cauchy', f_scale=.025, max_nfev=75).x
    order = np.argsort(v[:2])
    return np.exp(v[:2][order]), np.r_[v[2:4][order], v[4:]]


# The chamber resets the hereditary state at deployment. A constant forcing
# feeds the two relaxing branches and a remaining linear drift.
_CHAMBER = np.array([.20, .52, .28])


def _chamber(test, x, tau, theta, times, gains=_CHAMBER):
    force = x @ theta[2:]
    hold = np.asarray(test['hold_channel_raw']).sum(axis=-1)
    amplitudes = force[:, None] * gains * np.r_[tau, 1.]
    amplitudes[:, :2] += hold[:, None] * theta[:2]
    b, r = _basis(tau, times)
    return test['strain_0'][:, None] + amplitudes @ b.T, amplitudes @ r


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt, times = np.asarray(cal_item['past_dt']), np.asarray(cal_item['eval_times'])
    x_train, x_test = _features(train, test)
    tau, theta = _material(train, x_train, dt, times)
    strain, rate = _chamber(test, x_test, tau, theta, times)
    return dict(estimated_taus=tau, hereditary_weights=theta,
                predicted_strain=strain, initial_rate_pred=rate)
