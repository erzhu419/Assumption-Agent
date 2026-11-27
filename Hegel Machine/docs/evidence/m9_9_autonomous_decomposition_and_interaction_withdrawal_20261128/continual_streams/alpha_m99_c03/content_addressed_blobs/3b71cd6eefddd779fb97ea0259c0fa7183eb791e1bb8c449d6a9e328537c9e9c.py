import numpy as np
from scipy.optimize import least_squares


def _features(*blocks):
    x = np.vstack([b['coupon_features_raw'] for b in blocks]).astype(float)
    ref = np.vstack([b['ref_coupons'] for b in blocks])
    lab = np.concatenate([b['lab_id'] for b in blocks])
    stratum = np.concatenate([b['stratum'] for b in blocks])
    for group in np.unique(lab):
        mask = lab == group
        panel = ref[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = 1.0 / count[cell].reshape(-1, 1)
    observed = np.isfinite(x)
    mean = np.nansum(weight * x, axis=0) / np.sum(weight * observed, axis=0)
    return np.where(observed, x, mean)


def _design(taus, coupons, dt, times):
    past, hold, drift = coupons
    gap = _gap(taus, past, hold, dt)
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-times[:, None] / taus),
                            drift[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _fit(coupons, dt, times, curve_obs, rate_obs):
    curve_ok = np.isfinite(curve_obs.ravel())
    rate_ok = np.isfinite(rate_obs)
    curve_y = curve_obs.ravel()[curve_ok]
    rate_y = rate_obs[rate_ok]
    y = np.r_[curve_y, rate_y]
    ncurve = len(curve_y)

    def build(taus):
        curve, rate = _design(taus, coupons, dt, times)
        return np.vstack((curve.reshape(-1, 7)[curve_ok], rate[rate_ok]))

    def decode(p):
        fast = np.exp(p[0])
        return np.array((fast, fast + (25.0 - fast) * p[1]))

    def encode(fast, slow):
        return np.array((np.log(fast), (slow-fast)/(25.0-fast)))

    bounds = ([np.log(0.050001), 1e-6], [np.log(24.99), 1-1e-6])
    def profile(p):
        a = build(decode(p))
        return a @ np.linalg.lstsq(a, y, rcond=None)[0] - y

    grid = [encode(f, s) for f in (.08, .18, .4, .8, 1.6, 3.2)
            for s in (.3, .65, 1.3, 2.6, 5.2, 10.4, 21.) if f < s]
    starts = sorted(grid, key=lambda p: np.linalg.norm(profile(p)))[:3]
    ordinary = min((least_squares(profile, p, bounds=bounds, max_nfev=100)
                    for p in starts), key=lambda r: r.cost)
    a = build(decode(ordinary.x))
    theta = np.linalg.lstsq(a, y, rcond=None)[0]

    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))

    def scale(r):
        return np.r_[np.full(ncurve, mad(r[:ncurve])),
                     np.full(len(rate_y), mad(r[ncurve:]))]

    def robust(p, sigma, loss, f_scale):
        return least_squares(lambda v: (build(decode(v[:2])) @ v[2:] - y) / sigma,
                             p, bounds=(np.r_[bounds[0], [-np.inf]*7],
                                        np.r_[bounds[1], [np.inf]*7]),
                             loss=loss, f_scale=f_scale, x_scale='jac', max_nfev=200)

    sigma = scale(ordinary.fun)
    first = robust(np.r_[ordinary.x, theta], sigma, 'soft_l1', 1.5)
    sigma = scale(first.fun * sigma)
    result = robust(first.x, sigma, 'cauchy', 2.5)
    return decode(result.x[:2]), result.x[2:]


def _gap(taus, past, hold, dt):
    age = np.cumsum(dt[::-1])[::-1] - dt
    history = -np.expm1(-dt[:, None] / taus) * np.exp(-age[:, None] / taus)
    return hold[:, None] - past @ history


def _inputs(block, features):
    return (np.sum(block['past_channel_raw'], axis=-1),
            np.sum(block['hold_channel_raw'], axis=-1),
            np.column_stack((np.ones(len(features)), features)))


def _predict(block, features, taus, theta, dt, times):
    past, hold, features = _inputs(block, features)
    amplitude = _gap(taus, past, hold, dt) * theta[:2]
    drift = features @ theta[2:]
    relaxation = -np.expm1(-times[:, None] / taus)
    # The chamber drift rate decays with a one-second time constant.
    drift_kernel = -np.expm1(-times)
    strain = block['strain_0'][:, None] + amplitude @ relaxation.T + drift[:, None] * drift_kernel
    initial_rate = amplitude @ (1 / taus) + drift
    return strain, initial_rate


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt, times = np.asarray(cal_item['past_dt']), np.asarray(cal_item['eval_times'])
    # The intrinsic fit uses the joint benchtop/deployment reference protocol.
    x_train = _features(train, test)[:len(train['lab_id'])]
    taus, theta = _fit(_inputs(train, x_train), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The chamber drift uses the reference panel accompanying deployment.
    strain, rate = _predict(test, _features(test), taus, theta, dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
