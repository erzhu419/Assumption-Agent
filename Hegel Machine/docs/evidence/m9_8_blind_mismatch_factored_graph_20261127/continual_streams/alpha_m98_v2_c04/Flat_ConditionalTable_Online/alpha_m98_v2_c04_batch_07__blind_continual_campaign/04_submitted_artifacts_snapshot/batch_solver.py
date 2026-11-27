"""Calibrate the load bridge and fit a two-branch hereditary material law."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    for group in np.unique(lab):
        rows = lab == group
        panel = refs[rows]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1.0 / counts[cell, None]
    present = ~np.isnan(x)
    mass = np.sum(w * present, axis=0)
    fill = np.divide(np.nansum(w * x, axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    return np.where(present, x, fill)


def _bridge_basis(loads):
    nxt = np.roll(loads, -1, axis=1)
    prev = np.roll(loads, 1, axis=1)
    return np.stack((np.ones_like(loads), loads, nxt, prev,
                     loads * nxt, loads**3), axis=-1)


def _bridge(cal):
    loads = cal['true_channel_loads']
    raw = cal['bridge_raw']
    a = _bridge_basis(loads).transpose(1, 0, 2)
    w = np.ones(len(raw))
    for _ in range(25):
        b = np.stack([np.linalg.lstsq(aj * w[:, None], y * w, rcond=None)[0]
                      for aj, y in zip(a, raw.T)])
        e = np.linalg.norm(np.einsum('cnp,cp->nc', a, b) - raw, axis=1)
        w = np.sqrt(np.minimum(1., max(.00002, .45 * np.median(e)) /
                               np.maximum(e, 1e-15)))
    keep = e < max(.002, 5 * np.median(e))
    return np.stack([np.linalg.lstsq(aj[keep], y[keep], rcond=None)[0]
                     for aj, y in zip(a, raw.T)])


def _invert(raw, b):
    shape = raw.shape
    raw = raw.reshape(-1, 4)
    i = np.arange(4)
    j = (i + 1) % 4
    k = (i - 1) % 4
    mat = np.zeros((4, 4))
    mat[i, i], mat[i, j], mat[i, k] = b[:, 1], b[:, 2], b[:, 3]
    load = np.linalg.solve(mat, (raw - b[:, 0]).T).T
    jac = np.broadcast_to(mat, (len(load), 4, 4)).copy()
    for _ in range(20):
        residual = np.einsum('ncp,cp->nc', _bridge_basis(load), b) - raw
        jac[:, i, i] = b[:, 1] + b[:, 4] * load[:, j] + 3 * b[:, 5] * load**2
        jac[:, i, j] = b[:, 2] + b[:, 4] * load
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        load -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    return load.reshape(shape)


def _loads(block, bridge):
    return tuple(_invert(block[key], bridge).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(x, past, hold, dt, times, tau):
    # Weight past loads by their elapsed time to the start of the hold.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = -np.expm1(-dt[:, None] / tau) * np.exp(-later[:, None] / tau)
    gap = hold[:, None] - past @ memory
    creep = -np.expm1(-times[:, None] / tau)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * creep[None, :, :],
                            times[None, :, None] * drift[:, None, :]), axis=2)
    rate = np.column_stack((gap / tau, drift))
    return curve.reshape(-1, 7), rate


def _mad(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def _fit(x, past, hold, dt, times, curve_obs, rate_obs):
    yc = curve_obs.ravel()
    yr = rate_obs
    target = np.r_[yc, yr]
    lo, hi = np.log([0.05000001, 24.99999999])
    bounds = (np.r_[[lo, lo], np.full(7, -np.inf)],
              np.r_[[hi, hi], np.full(7, np.inf)])

    def design(logtau):
        a, b = _design(x, past, hold, dt, times, np.exp(logtau))
        return np.vstack((a, b))

    def residual(p):
        return (design(p[:2]) @ p[2:] - target) / scale

    scale = max(0.01, 0.015 * np.std(yc))
    trials = []
    for guess in ((0.18, 1.6), (0.45, 4.0), (0.9, 12.0), (0.12, 8.0)):
        q = np.log(guess)
        coeff = np.linalg.lstsq(design(q), target, rcond=None)[0]
        trials.append(least_squares(residual, np.r_[q, coeff], bounds=bounds,
                                    loss='soft_l1', x_scale='jac', max_nfev=180))
    p = min(trials, key=lambda r: r.cost).x
    for _ in range(2):
        e = design(p[:2]) @ p[2:] - target
        sc = max(_mad(e[:len(yc)]), 1e-6)
        sr = max(_mad(e[len(yc):]), 1e-6)
        # Cauchy downweighting preserves the Gaussian center of each sensor.
        scale = 2.5 * np.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]


def _chamber(x, past, hold, dt, times, taus, theta):
    # Keep the intrinsic memory and relaxation times. Only the chamber
    # drift saturates, with unit time scale in the supplied time units.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = -np.expm1(-dt[:, None] / taus) * np.exp(-later[:, None] / taus)
    amplitudes = (hold[:, None] - past @ memory) * theta[:2]
    drift = theta[2] + x @ theta[3:]
    creep = -np.expm1(-times[:, None] / taus)
    drift_curve = -np.expm1(-times)
    return (amplitudes @ creep.T + drift[:, None] * drift_curve,
            amplitudes @ (1.0 / taus) + drift)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(x[:n], *_loads(train, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    change, rate = _chamber(x[n:], *_loads(test, bridge), dt, times, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + change,
            'initial_rate_pred': rate}
