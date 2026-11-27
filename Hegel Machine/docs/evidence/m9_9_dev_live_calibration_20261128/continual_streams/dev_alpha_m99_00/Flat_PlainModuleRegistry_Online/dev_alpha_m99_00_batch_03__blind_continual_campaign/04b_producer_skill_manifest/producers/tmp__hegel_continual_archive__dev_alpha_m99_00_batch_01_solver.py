"""Calibrate the load bridge and fit the two-branch hereditary model."""
import numpy as np
from scipy.optimize import least_squares


def _bridge(cal):
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    return np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))),
                           loads.sum(axis=1), rcond=None)[0]


def _features(*splits):
    x = np.concatenate([p['coupon_features_raw'] for p in splits]).astype(float)
    refs = np.concatenate([p['ref_coupons'] for p in splits])
    lab = np.concatenate([p['lab_id'] for p in splits])
    strata = np.concatenate([p['stratum'] for p in splits])
    for label in np.unique(lab):
        rows = lab == label
        panel = refs[rows]
        x[rows] = (x[rows] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, count = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = 1.0 / count[cell]
    valid = np.isfinite(x)
    denominator = (valid * weight[:, None]).sum(axis=0)
    means = np.divide(weight @ np.where(valid, x, 0.0), denominator,
                      out=np.zeros(x.shape[1]), where=denominator != 0)
    return np.where(valid, x, means)


def _coupon(split, x, bridge):
    past = split['past_channel_raw'] @ bridge[:-1] + bridge[-1]
    hold = split['hold_channel_raw'] @ bridge[:-1] + bridge[-1]
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(taus, coupon, dt, t):
    past, hold, drift = coupon
    z = np.zeros((len(hold), 2))
    for step, duration in enumerate(dt):
        uptake = -np.expm1(-duration / taus)
        z += uptake * (past[:, step, None] - z)
    gap = hold[:, None] - z
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            drift[:, None, :] * t[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _fit(coupon, dt, t, curve_obs, rate_obs):
    n, k = curve_obs.shape
    y = np.r_[curve_obs.ravel(), rate_obs]
    lower, upper = np.log([0.05000001, 24.999999])
    outer_bounds = (np.full(2, lower), np.full(2, upper))

    def design(log_tau):
        curve, rate = _design(np.exp(log_tau), coupon, dt, t)
        return np.vstack((curve.reshape(-1, 7), rate))

    def profile(log_tau):
        a = design(log_tau)
        return a @ np.linalg.lstsq(a, y, rcond=None)[0] - y

    starts = ((0.18, 1.7), (0.45, 4.5), (0.9, 12.0), (2.0, 18.0))
    pilot = min((least_squares(profile, np.log(s), bounds=outer_bounds,
                               max_nfev=55) for s in starts), key=lambda r: r.cost)
    a = design(pilot.x)
    theta = np.linalg.lstsq(a, y, rcond=None)[0]

    def scales(resid):
        curve, rate = resid[:n*k], resid[n*k:]
        return np.repeat([max(1e-7, 1.4826 * np.median(np.abs(e - np.median(e))))
                          for e in (curve, rate)], [n*k, n])

    bounds = (np.r_[outer_bounds[0], [-np.inf]*7], np.r_[outer_bounds[1], [np.inf]*7])
    p = np.r_[pilot.x, theta]
    for loss, cutoff in (('soft_l1', 1.7), ('cauchy', 3.0)):
        scale = scales(design(p[:2]) @ p[2:] - y)
        p = least_squares(lambda v: (design(v[:2]) @ v[2:] - y) / scale,
                          p, bounds=bounds, loss=loss, f_scale=cutoff,
                          x_scale='jac', max_nfev=125).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def _chamber(taus, theta, hold, drift, times):
    # Equilibration resets both memory branches before the deployment hold.
    # Integrating the chamber drift rate r/(1+t) gives r*log(1+t).
    response = -np.expm1(-times[:, None] / taus) @ theta[:2]
    strain_change = hold[:, None] * response + drift[:, None] * np.log1p(times)
    initial_rate = hold * (theta[:2] @ (1.0 / taus)) + drift
    return strain_change, initial_rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(_coupon(train, x[:n], bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    hold = test['hold_channel_raw'] @ bridge[:-1] + bridge[-1]
    drift = theta[2] + x[n:] @ theta[3:]
    strain, rate = _chamber(taus, theta, hold, drift, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
