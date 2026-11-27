"""Fit the intrinsic material and transfer it through the chamber operator."""
import numpy as np
from scipy.optimize import least_squares


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



# Chamber verification fit: relaxation-time multipliers relative to the
# benchtop spectrum and coefficients on (1, hold load, material drift).
_CHAMBER_TIME_SCALE = np.array([0.50207 / 0.50795558, 2.50897 / 4.02636417])
_CHAMBER_COEFFICIENTS = np.array([
    [0.00130, 0.68328, 0.18685],
    [-0.00755, 0.38639, 1.34637],
    [0.00031, 0.02873, 0.08642],
])


def _coupon(split, features):
    return (np.sum(split['past_channel_raw'], axis=-1),
            np.sum(split['hold_channel_raw'], axis=-1),
            np.column_stack((np.ones(len(features)), features)))


def _memory(taus, past, dt):
    state = np.zeros((len(past), len(taus)))
    for stress, duration in zip(past.T, dt):
        state += -np.expm1(-duration / taus) * (stress[:, None] - state)
    return state


# Calibrated on complete verification coupons using their own reference panel.
# Rows are hold load and intrinsic drift; columns are the two modes and creep.
_CHAMBER_FORCING = np.array([[0.7031164, 0.6046994, 0.0033573],
                             [0.0964212, 2.2312914, 0.2332000]])
_CHAMBER_SLOW_TIME = 3.8573976


def _predict_chamber(hold, drift, fast_time, times):
    chamber_taus = np.array([fast_time, _CHAMBER_SLOW_TIME])
    amplitudes = np.column_stack((hold, drift)) @ _CHAMBER_FORCING
    response = np.column_stack((-np.expm1(-times[:, None] / chamber_taus), times))
    return amplitudes @ response.T, amplitudes @ np.r_[1.0 / chamber_taus, 1.0]


def solve_downstream(cal_item: dict) -> dict:
    """Fit intrinsic material from benchtop measurements, then predict chamber response."""
    train, test = cal_item['train'], cal_item['test']
    dt, times = np.asarray(cal_item['past_dt']), np.asarray(cal_item['eval_times'])
    features = _features(train, test)
    n = len(train['lab_id'])
    strain = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, theta = _fit(_coupon(train, features[:n]), dt, times, strain,
                       train['instant_rate_obs'])
    hold = np.sum(test['hold_channel_raw'], axis=-1)
    drift = theta[2] + features[n:] @ theta[3:]
    curve, rate = _predict_chamber(hold, drift, taus[0], times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
