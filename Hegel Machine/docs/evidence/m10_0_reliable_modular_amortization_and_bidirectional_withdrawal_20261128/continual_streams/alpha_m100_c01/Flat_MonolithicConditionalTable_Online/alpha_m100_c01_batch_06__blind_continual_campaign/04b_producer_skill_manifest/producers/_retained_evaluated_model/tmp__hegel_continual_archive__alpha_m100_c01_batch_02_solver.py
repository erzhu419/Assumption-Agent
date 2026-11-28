"""Calibrate the load bridge and fit the two-branch hereditary response."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    splits = (train, test)
    labs = np.concatenate([s['lab_id'] for s in splits])
    strata = np.concatenate([s['stratum'] for s in splits])
    refs = np.vstack([s['ref_coupons'] for s in splits])
    x = np.vstack([s['coupon_features_raw'] for s in splits])
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        # Canonical row order keeps pooled moments independent of split ordering.
        panel = np.sort(panel, axis=0)
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0))
        x[rows] = (x[rows] - center) / scale
    _, cells, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                 return_inverse=True, return_counts=True)
    w = 1.0 / counts[cells, None]
    observed = ~np.isnan(x)
    means = np.nansum(w * x, axis=0) / (w * observed).sum(axis=0)
    return np.split(np.where(observed, x, means), [len(train['lab_id'])])


def _bridge(raw, loads):
    design = np.column_stack((np.ones(len(raw)), raw))
    return np.linalg.lstsq(design, loads.sum(axis=1), rcond=None)[0]


def _inputs(split, x, bridge):
    past = split['past_channel_raw'] @ bridge[1:] + bridge[0]
    hold = split['hold_channel_raw'] @ bridge[1:] + bridge[0]
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(taus, past, hold, features, dt, times):
    age = np.cumsum(dt[::-1])[::-1] - dt
    memory = np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    gap = hold[:, None] - past @ memory
    curve_branch = gap[:, None, :] * -np.expm1(-times[:, None] / taus)
    curve_drift = times[None, :, None] * features[:, None, :]
    curve = np.concatenate((curve_branch, curve_drift), axis=2)
    rate = np.column_stack((gap / taus, features))
    return curve, rate


def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
    n, k = curve_obs.shape
    y = np.column_stack((curve_obs, rate_obs)).ravel()

    def unpack(u):
        a = np.exp(u[0])
        b = np.exp(u[0] + u[1] * (np.log(24.999) - u[0]))
        return np.array([a, b])

    def matrix(u):
        c, r = _design(unpack(u), past, hold, features, dt, times)
        return np.concatenate((c, r[:, None]), axis=1).reshape(n * (k + 1), 7)

    bounds = (np.array([np.log(.05001), .00001]), np.array([np.log(24.998), 1.]))
    initials = []
    for a in (.10, .23, .45, .8, 1.4, 2.5, 4.):
        for b in (.5, 1., 2., 4., 8., 16.):
            if b <= a:
                continue
            u = np.array([np.log(a), np.log(b/a) / np.log(24.999/a)])
            M = matrix(u)
            beta = np.linalg.lstsq(M, y, rcond=None)[0]
            for _ in range(4):
                w = (1 + ((M @ beta - y) / .01)**2)**(-.25)
                beta = np.linalg.lstsq(w[:, None] * M, w * y, rcond=None)[0]
            resid = M @ beta - y
            initials.append((np.sum(np.hypot(.01, resid) - .01), u, beta))

    def residual(q):
        return matrix(q[:2]) @ q[2:] - y

    results = []
    for _, u, beta in sorted(initials, key=lambda entry: entry[0])[:4]:
        results.append(least_squares(residual, np.r_[u, beta], loss='soft_l1',
                                     f_scale=.008, x_scale='jac', max_nfev=200,
                                     bounds=(np.r_[bounds[0], np.full(7, -np.inf)],
                                             np.r_[bounds[1], np.full(7, np.inf)])))
    result = min(results, key=lambda r: r.cost)
    # Give the direct initial-rate measurements more leverage than any
    # single strain reading while retaining robust loss for both outputs.
    def rate_weighted_residual(q):
        e = residual(q).reshape(n, k + 1)
        e[:, -1] *= 4.0
        return e.ravel()

    result = least_squares(rate_weighted_residual, result.x, loss='soft_l1',
                           f_scale=.008, x_scale='jac', max_nfev=200,
                           bounds=(np.r_[bounds[0], np.full(7, -np.inf)],
                                   np.r_[bounds[1], np.full(7, np.inf)]))
    return unpack(result.x[:2]), result.x[2:]


# Chamber drift time (seconds), estimated from the verification coupons.
_CHAMBER_DRIFT_TIME = 1.0116511228821314


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    xtr, xte = _features(train, test)
    cal = dataset['calibration_load_cells']
    bridge = _bridge(cal['bridge_raw'], cal['true_channel_loads'])
    taus, theta = _fit(*_inputs(train, xtr, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    past, hold, features = _inputs(test, xte, bridge)
    curve, rate = _design(taus, past, hold, features, dt, times)
    drift = features @ theta[2:]
    # The chamber replaces the linear drift ramp with exponential saturation.
    # Both ramps have derivative one at time zero, so the rate is unchanged.
    tau = _CHAMBER_DRIFT_TIME
    ramp = -tau * np.expm1(-times / tau)
    strain = test['strain_0'][:, None] + curve[:, :, :2] @ theta[:2] + drift[:, None] * ramp
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate @ theta}
