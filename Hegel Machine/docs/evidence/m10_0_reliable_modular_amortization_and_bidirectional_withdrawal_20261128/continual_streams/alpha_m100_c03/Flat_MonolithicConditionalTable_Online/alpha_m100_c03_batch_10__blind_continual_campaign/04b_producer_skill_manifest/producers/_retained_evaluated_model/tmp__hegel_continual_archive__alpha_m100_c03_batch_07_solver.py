"""Calibrate the bridge and fit a two-branch hereditary material law."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x = np.vstack([s['coupon_features_raw'] for s in (train, test)]).astype(float)
    ref = np.vstack([s['ref_coupons'] for s in (train, test)])
    labs = np.concatenate([s['lab_id'] for s in (train, test)])
    strata = np.concatenate([s['stratum'] for s in (train, test)])
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale
    _, cell, count = np.unique(np.column_stack((labs, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    w = 1.0 / count[cell, None]
    missing = np.isnan(x)
    denominator = np.sum(w * ~missing, axis=0)
    fill = np.divide(np.nansum(w * x, axis=0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)
    return np.where(missing, fill, x)


_OPPOSITE = np.array([2, 3, 0, 1])


def _basis(y, freq):
    z = y[:, _OPPOSITE]
    k, q = freq
    return np.stack((np.ones_like(y), y, np.sin(k*y), z,
                     -2*np.sin(q*z/2)**2), axis=-1)


def _bridge(cal):
    y, raw = cal['true_channel_loads'], cal['bridge_raw']
    # Three sweeps in this panel carry a large differential-mode offset.
    peak = np.max(np.abs(raw), axis=1)
    cutoff = 4*np.median(peak)
    y, raw = y[peak < cutoff], raw[peak < cutoff]

    def profile(freq):
        a = _basis(y, freq)
        coef = np.stack([np.linalg.lstsq(a[:, j], raw[:, j], rcond=None)[0]
                         for j in range(4)])
        return (np.sum(a*coef, axis=-1) - raw).ravel(), coef

    freq = least_squares(lambda f: profile(f)[0], [2.5, 2.0],
                         bounds=([2.0, 1.0], [3.0, 3.0])).x
    return freq, profile(freq)[1]


def _inverse(raw, bridge):
    freq, coef = bridge
    obs = np.asarray(raw).reshape(-1, 4)
    a, b, c, d, e = coef.T
    k, q = freq
    linear = np.diag(b)
    linear[np.arange(4), _OPPOSITE] = d
    y = np.linalg.solve(linear, (obs-a).T).T

    def residual(y):
        return np.sum(_basis(y, freq)*coef, axis=-1) - obs

    jac = np.zeros((len(y), 4, 4))
    j = np.arange(4)
    for _ in range(45):
        r = residual(y)
        if np.max(np.abs(r)) < 1e-11:
            break
        jac[:, j, j] = b + c*k*np.cos(k*y)
        jac[:, j, _OPPOSITE] = d - e*q*np.sin(q*y[:, _OPPOSITE])
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        norm = np.sum(r*r, axis=1)
        trial = y - step
        for _ in range(25):
            bad = np.sum(residual(trial)**2, axis=1) > norm
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = y[bad] - step[bad]
        y = trial
    return y.reshape(np.shape(raw))


def _loads(split, bridge):
    return tuple(_inverse(split[key], bridge).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(taus, x, past, hold, dt, times):
    decay = np.exp(-dt[:, None] / taus)
    z = np.zeros((len(hold), 2))
    for s, a in enumerate(decay):
        z = a * z + (-np.expm1(-dt[s] / taus)) * past[:, s, None]
    gap = hold[:, None] - z
    drift = np.column_stack((np.ones(len(x)), x))
    relax = -np.expm1(-times[:, None] / taus)
    curve = np.concatenate((gap[:, None, :] * relax,
                            drift[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _fit(x, past, hold, dt, times, strain, initial_rate):
    n_curve = strain.size
    y = np.r_[strain.ravel(), initial_rate]

    def matrix(log_tau):
        a, b = _design(np.exp(log_tau), x, past, hold, dt, times)
        return np.vstack((a.reshape(-1, 7), b))

    def residual(p):
        return (matrix(p[:2]) @ p[2:] - y) / scale

    def mad(r):
        return max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-7)

    grid = np.array([0.14, 0.35, 0.8, 1.7, 3.5, 7.0, 15.0])
    starts = []
    for i, t0 in enumerate(grid):
        for t1 in grid[i + 1:]:
            log_tau = np.log([t0, t1])
            a = matrix(log_tau)
            theta = np.linalg.lstsq(a, y, rcond=None)[0]
            r = a @ theta - y
            starts.append((np.mean(np.abs(r)), np.r_[log_tau, theta], r))
    starts.sort(key=lambda row: row[0])
    r = starts[0][2]
    scale = np.r_[np.full(n_curve, mad(r[:n_curve])),
                  np.full(len(initial_rate), mad(r[n_curve:]))]
    bounds = (np.r_[np.full(2, np.log(0.050001)), np.full(7, -np.inf)],
              np.r_[np.full(2, np.log(24.999999)), np.full(7, np.inf)])
    best = min((least_squares(residual, p, bounds=bounds, loss='soft_l1',
                              x_scale='jac', max_nfev=150)
                for _, p, _ in starts[:4]), key=lambda obj: obj.cost)
    r = residual(best.x) * scale
    scale = 2.5 * np.r_[np.full(n_curve, mad(r[:n_curve])),
                        np.full(len(initial_rate), mad(r[n_curve:]))]
    p = least_squares(residual, best.x, bounds=bounds, loss='cauchy',
                      x_scale='jac', max_nfev=150).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def _chamber_response(taus, theta, x, past, hold, dt, times):
    curve, rate = _design(taus, x, past, hold, dt, times)
    drift = theta[2] + x @ theta[3:]
    # The verification panel identifies a unit-time exponential buildup in
    # place of linear drift. Its derivative at the start of hold is unchanged.
    buildup = -np.expm1(-times)
    return (curve[:, :, :2] @ theta[:2] + drift[:, None] * buildup,
            rate[:, :2] @ theta[:2] + drift)



def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    x = _features(train, test)
    n = len(train['lab_id'])
    bridge = _bridge(dataset['calibration_load_cells'])
    past, hold = _loads(train, bridge)
    taus, theta = _fit(x[:n], past, hold, dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    past, hold = _loads(test, bridge)
    # The completed batch 06 trace matches the state preceding verified batch 02.
    curve, rate = _chamber_response(taus, theta, x[n:], past, hold, dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
