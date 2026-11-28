"""Fit the intrinsic material law and predict calibrated chamber responses."""

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


def _loads(split):
    return tuple(np.asarray(split[key], dtype=float).sum(axis=-1)
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



def _predict_chamber(taus, theta, x, hold, times, strain_0):
    """Reset the branch memory and route the coupon drift into both branches."""
    drift = theta[2] + x @ theta[3:]
    amplitude = hold[:, None] * theta[:2] + 0.5 * drift[:, None] * taus
    strain = strain_0[:, None] + amplitude @ (-np.expm1(-times[:, None] / taus)).T
    rate = amplitude @ (1.0 / taus)
    return strain, rate


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt = np.asarray(cal_item['past_dt'], dtype=float)
    times = np.asarray(cal_item['eval_times'], dtype=float)
    x = _features(train, test)
    n = len(train['lab_id'])
    past, hold = _loads(train)
    taus, theta = _fit(x[:n], past, hold, dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    hold = np.asarray(test['hold_channel_raw'], dtype=float).sum(axis=-1)
    strain, rate = _predict_chamber(taus, theta, x[n:], hold, times, test['strain_0'])
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
