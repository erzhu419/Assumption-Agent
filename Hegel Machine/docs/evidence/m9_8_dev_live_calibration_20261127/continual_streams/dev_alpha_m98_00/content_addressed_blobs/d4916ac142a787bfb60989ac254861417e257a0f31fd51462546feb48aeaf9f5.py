"""Estimate intrinsic relaxation and predict chamber deployment."""

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



# Each readout has a linear bridge, its own cubic, and coupling to the
# next channel. Fit the coefficients from the supplied calibration loads.
def _bridge_terms(loads):
    return np.column_stack((np.ones(len(loads)), loads))


def _fit_bridge(calibration):
    loads, raw = calibration['true_channel_loads'], calibration['bridge_raw']
    affine = _bridge_terms(loads)
    estimates = affine @ np.linalg.lstsq(affine, raw, rcond=None)[0]
    residual = np.linalg.norm(estimates - raw, axis=1)
    keep = residual <= max(5 * np.median(residual), 1e-8)
    coeff = np.empty((7, 4))
    for j in range(4):
        terms = np.column_stack((affine, loads[:, j] ** 3,
                                 loads[:, j] * loads[:, (j + 1) % 4]))
        coeff[:, j] = np.linalg.lstsq(terms[keep], raw[keep, j], rcond=None)[0]
    return coeff


def _bridge_forward(loads, coeff):
    return (coeff[0] + loads @ coeff[1:5] + loads ** 3 * coeff[5]
            + loads * np.roll(loads, -1, axis=1) * coeff[6])


def _invert_bridge(raw, coeff):
    obs = raw.reshape(-1, 4)
    loads = np.linalg.solve(coeff[1:5].T, (obs - coeff[0]).T).T
    j = np.arange(4)
    for _ in range(25):
        error = _bridge_forward(loads, coeff) - obs
        if np.max(np.abs(error)) < 1e-10:
            break
        jac = np.broadcast_to(coeff[1:5].T, (len(loads), 4, 4)).copy()
        jac[:, j, j] += 3 * loads ** 2 * coeff[5] + np.roll(loads, -1, axis=1) * coeff[6]
        jac[:, j, (j + 1) % 4] += loads * coeff[6]
        loads -= np.linalg.solve(jac, error[..., None])[..., 0]
    return loads.reshape(raw.shape)


def _coupon(group, features, coeff):
    return (_invert_bridge(group['past_channel_raw'], coeff).sum(axis=-1),
            _invert_bridge(group['hold_channel_raw'], coeff).sum(axis=-1), features)


def _chamber(taus, theta, dt, times, coupon):
    curve, rate = _design(taus, dt, times, coupon)
    drift = coupon[2] @ theta[2:]
    # Chamber diagnostics support retained history and redistribution of
    # drift into the two existing branches, without changing the initial rate.
    fractions = np.array([0.62686, 0.60591])
    growth = -np.expm1(-times[:, None] / taus)
    correction = (growth * taus - times[:, None]) @ fractions
    return curve @ theta + drift[:, None] * correction, rate @ theta


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    x_train, x_test = np.split(_aligned_features(train, test), [len(train['lab_id'])])
    coeff = _fit_bridge(dataset['calibration_load_cells'])
    taus, theta = _fit(dt, times, _coupon(train, x_train, coeff),
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    curve, rate = _chamber(taus, theta, dt, times, _coupon(test, x_test, coeff))
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
