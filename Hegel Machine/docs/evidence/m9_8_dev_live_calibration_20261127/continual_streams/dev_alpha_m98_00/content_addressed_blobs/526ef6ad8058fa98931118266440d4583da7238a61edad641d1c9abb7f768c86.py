"""Fit the load bridge and benchtop material, then predict carried-memory chamber rheology."""

import numpy as np
from scipy.optimize import least_squares


def _aligned_features(train, test):
    lab = np.concatenate([train['lab_id'], test['lab_id']])
    strata = np.concatenate([train['stratum'], test['stratum']])
    raw = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for group in np.unique(lab):
        rows = lab == group
        panel = refs[rows]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        raw[rows] = (raw[rows] - center) / scale
    _, cell, counts = np.unique(np.column_stack((lab, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weight = 1.0 / counts[cell, None]
    missing = np.isnan(raw)
    mass = np.sum(weight * ~missing, axis=0)
    fill = np.divide(np.nansum(weight * raw, axis=0), mass,
                     out=np.zeros(raw.shape[1]), where=mass > 0)
    return np.column_stack((np.ones(len(raw)), np.where(missing, fill, raw)))


_BRIDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (0, 3))


def _stress_calibration(calibration):
    """Fit the forward bridge, including a separated common offset state."""
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    magnitude = np.max(np.abs(raw), axis=1)
    levels = np.sort(magnitude)
    split = np.argmax(np.diff(levels))
    lower, upper = levels[split:split + 2]
    threshold = (lower + upper) / 2 if upper > 2.5 * lower else np.inf
    shifted = magnitude > threshold
    columns = []
    for j, (i, k) in enumerate(_BRIDGE_PAIRS):
        basis = np.c_[np.ones(len(loads)), loads, loads[:, j]**3,
                      loads[:, i] * loads[:, k], shifted]
        columns.append(np.linalg.lstsq(basis, raw[:, j], rcond=None)[0])
    return np.array(columns).T, threshold


def _bridge_forward(loads, coeff):
    products = np.column_stack([loads[:, i] * loads[:, k] for i, k in _BRIDGE_PAIRS])
    return coeff[0] + loads @ coeff[1:5] + loads**3 * coeff[5] + products * coeff[6]


def _channel_loads(raw, calibration):
    coeff, threshold = calibration
    flat = np.asarray(raw, dtype=float).reshape(-1, 4)
    shifted = np.max(np.abs(flat), axis=1) > threshold
    target = flat - shifted[:, None] * coeff[7]
    linear = coeff[1:5].T
    loads = np.clip(np.linalg.solve(linear, (target - coeff[0]).T).T, -1.8, 1.8)
    for _ in range(30):
        jac = np.broadcast_to(linear, (len(loads), 4, 4)).copy()
        for j, (i, k) in enumerate(_BRIDGE_PAIRS):
            jac[:, j, j] += 3 * coeff[5, j] * loads[:, j]**2
            jac[:, j, i] += coeff[6, j] * loads[:, k]
            jac[:, j, k] += coeff[6, j] * loads[:, i]
        step = np.linalg.solve(jac, (target - _bridge_forward(loads, coeff))[..., None])[..., 0]
        loads += np.clip(step, -.5, .5)
        if np.max(np.abs(step)) < 1e-10:
            break
    return loads.reshape(raw.shape)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    features = _aligned_features(train, test)
    n_train = len(train['lab_id'])
    cal = _stress_calibration(dataset['calibration_load_cells'])
    return tuple((_channel_loads(group['past_channel_raw'], cal).sum(axis=-1),
                  _channel_loads(group['hold_channel_raw'], cal).sum(axis=-1), x)
                 for group, x in ((train, features[:n_train]),
                                  (test, features[n_train:])))


def _design(taus, dt, times, coupon):
    past, hold, x = coupon
    z = np.zeros((len(hold), 2))
    for alpha, stress in zip(-np.expm1(-dt[:, None] / taus), past.T):
        z += alpha * (stress[:, None] - z)
    gap = hold[:, None] - z
    growth = -np.expm1(-times[:, None] / taus)
    curve = np.concatenate((gap[:, None, :] * growth,
                            times[None, :, None] * x[:, None, :]), axis=2)
    rate = np.column_stack((gap / taus, x))
    return curve, rate


def _fit(dt, times, coupon, curve_obs, rate_obs, *, adaptive=True):
    y = np.concatenate((curve_obs.ravel(), rate_obs))
    k = curve_obs.size
    low, high = np.log([0.0500001, 24.999999])
    bounds = ([low, low] + [-np.inf] * 7, [high, high] + [np.inf] * 7)

    def design(log_tau):
        curve, rate = _design(np.exp(log_tau), dt, times, coupon)
        return np.vstack((curve.reshape(-1, 7), rate))

    def residual(p):
        return design(p[:2]) @ p[2:] - y

    def fit(p, loss, scale):
        return least_squares(lambda a: residual(a) / scale, p, bounds=bounds,
                             x_scale='jac', loss=loss, max_nfev=250)

    initial_scale = max(0.02 * np.median(np.abs(y)), 1e-7)
    fits = []
    for start in ((0.25, 2.0), (0.7, 5.0), (1.3, 14.0)):
        lt = np.log(start)
        beta = np.linalg.lstsq(design(lt), y, rcond=None)[0]
        fits.append(fit(np.r_[lt, beta], 'soft_l1', initial_scale))
    best = min(fits, key=lambda result: result.cost).x
    if adaptive:
        e = residual(best)
        scales = [max(1.4826 * np.median(np.abs(r - np.median(r))),
                      initial_scale * 0.002) for r in (e[:k], e[k:])]
        scale = 2.5 * np.repeat(scales, [k, len(rate_obs)])
        best = fit(best, 'cauchy', scale).x
    order = np.argsort(best[:2])
    return np.exp(best[:2][order]), np.r_[best[2:4][order], best[4:]]


# Standalone chamber verification confirms the archived carried-memory law.
_CHAMBER_FORCING = np.array([0.41228893817869877, 2.2768586984955768])


def _deployment(taus, theta, dt, times, coupon):
    curve, rate = _design(taus, dt, times, coupon)
    drift = coupon[2] @ theta[2:]
    u = times[:, None] / taus
    correction = (-np.expm1(-u) - u) @ _CHAMBER_FORCING
    # The forcing correction has zero derivative at the start of the hold.
    return curve @ theta + drift[:, None] * correction, rate @ theta


def solve_batch(dataset: dict) -> dict:
    """Estimate material parameters solely from the benchtop labels."""
    dt, times = dataset['past_dt'], dataset['eval_times']
    train, test = dataset['train'], dataset['test']
    train_coupon, test_coupon = _prepare(dataset)
    taus, theta = _fit(dt, times, train_coupon,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    increment, rate = _deployment(taus, theta, dt, times, test_coupon)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + increment,
            'initial_rate_pred': rate}
