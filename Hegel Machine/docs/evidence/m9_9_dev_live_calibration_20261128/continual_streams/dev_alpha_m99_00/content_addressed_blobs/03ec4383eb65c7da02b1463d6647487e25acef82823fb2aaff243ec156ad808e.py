"""Fit benchtop relaxation and predict the equilibrated chamber response."""
import numpy as np
from scipy.optimize import least_squares


def _features(*splits):
    x = np.concatenate([s['coupon_features_raw'] for s in splits]).astype(float)
    refs = np.concatenate([s['ref_coupons'] for s in splits])
    labs = np.concatenate([s['lab_id'] for s in splits])
    strata = np.concatenate([s['stratum'] for s in splits])
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, count = np.unique(np.column_stack((labs, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    w = 1.0 / count[cell]
    known = np.isfinite(x)
    means = (w @ np.where(known, x, 0.0)) / (w @ known)
    return np.where(known, x, means)


def _design(taus, past, hold, features, dt, times):
    state = np.zeros((len(hold), 2))
    for stress, duration in zip(past.T, dt):
        state += -np.expm1(-duration / taus) * (stress[:, None] - state)
    gap = hold[:, None] - state
    drift = np.column_stack((np.ones(len(hold)), features))
    curves = np.concatenate((gap[:, None, :] * -np.expm1(-times[:, None] / taus),
                             times[None, :, None] * drift[:, None, :]), axis=2)
    rates = np.column_stack((gap / taus, drift))
    return curves, rates


def _loads(split, projection):
    weights, offset = projection
    return (split['past_channel_raw'] @ weights + offset,
            split['hold_channel_raw'] @ weights + offset)


def _fit(train, features, dt, times, projection):
    past, hold = _loads(train, projection)
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    rates = train['instant_rate_obs']

    def residual(p):
        c, r = _design(p[:2], past, hold, features, dt, times)
        return np.r_[(c @ p[2:] - curves).ravel(), r @ p[2:] - rates]

    lo = [0.050001, 0.050001] + [-np.inf] * 7
    hi = [24.999999, 24.999999] + [np.inf] * 7
    fits = []
    for taus in ([0.4, 3.0], [1.0, 9.0], [0.12, 16.0]):
        c, r = _design(np.asarray(taus), past, hold, features, dt, times)
        theta = np.linalg.lstsq(np.vstack((c.reshape(-1, 7), r)),
                               np.r_[curves.ravel(), rates], rcond=None)[0]
        fits.append(least_squares(residual, np.r_[taus, theta], bounds=(lo, hi),
                                  loss='soft_l1', f_scale=0.004, x_scale='jac'))
    p = min(fits, key=lambda fit: fit.cost).x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def _predict_chamber(taus, theta, split, features, times, projection):
    hold = split['hold_channel_raw'] @ projection[0] + projection[1]
    drift = theta[2] + features @ theta[3:]
    response = -np.expm1(-times[:, None] / taus) @ theta[:2]
    # Both memory branches start at equilibrium; integrate r / sqrt(1+t).
    integral = 2 * times / (np.sqrt(1 + times) + 1)
    strain = split['strain_0'][:, None] + hold[:, None] * response + drift[:, None] * integral
    rate = hold * np.sum(theta[:2] / taus) + drift
    return strain, rate


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt = np.asarray(cal_item['past_dt'], dtype=float)
    times = np.asarray(cal_item['eval_times'], dtype=float)
    features = _features(train, test)
    n = len(train['strain_0'])
    projection = cal_item['total_load_projection']
    taus, theta = _fit(train, features[:n], dt, times, projection)
    strain, rate = _predict_chamber(taus, theta, test, features[n:], times, projection)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
