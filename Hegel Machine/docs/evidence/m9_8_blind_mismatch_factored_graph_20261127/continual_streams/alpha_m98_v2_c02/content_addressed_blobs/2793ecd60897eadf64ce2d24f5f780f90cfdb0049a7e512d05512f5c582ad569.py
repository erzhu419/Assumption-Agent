"""Fit benchtop material parameters and predict the chamber response."""

import numpy as np
from scipy.optimize import least_squares


def _standardize(train, test):
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0)
                            for k in ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for group in np.unique(lab):
        mask = lab == group
        controls = ref[mask]
        offset = np.nanmedian(np.nanmean(controls, axis=0))
        scale = np.nanmedian(np.nanstd(controls, axis=0))
        x[mask] = (x[mask] - offset) / scale
    _, inverse, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                  return_inverse=True, return_counts=True)
    weight = (1.0 / counts[inverse]).reshape(-1, 1)
    observed = np.isfinite(x)
    fill = (np.sum(weight * np.where(observed, x, 0), axis=0)
            / np.sum(weight * observed, axis=0))
    x = np.where(observed, x, fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _stress_calibration(calibration):
    """Fit the cyclic forward bridge and invert it to recover axial load."""
    loads = calibration['true_channel_loads']
    raw = calibration['bridge_raw']
    prev = np.roll(loads, 1, axis=-1)
    nxt = np.roll(loads, -1, axis=-1)
    a = np.stack((np.ones_like(loads), loads, prev, nxt, loads * nxt, loads**3), axis=-1)
    coefs = []
    for j in range(4):
        design = a[:, j, :]
        y = raw[:, j]
        b = np.linalg.lstsq(design, y, rcond=None)[0]
        for scale in (0.3, 0.01, 0.0003):
            b = least_squares(lambda c: design @ c - y, b, loss='soft_l1', f_scale=scale).x
        e = design @ b - y
        mad = np.median(np.abs(e - np.median(e)))
        keep = np.abs(e - np.median(e)) < max(8 * mad, 0.001)
        coefs.append(np.linalg.lstsq(design[keep], y[keep], rcond=None)[0])
    offset, gain, left, right, cross, cubic = np.asarray(coefs).T
    ids = np.arange(4)
    before, after = (ids - 1) % 4, (ids + 1) % 4
    linear = np.diag(gain)
    linear[ids, before] = left
    linear[ids, after] = right

    def stress(bridge):
        y = bridge - offset
        x = y @ np.linalg.inv(linear).T
        for _ in range(20):
            nxt = x[..., after]
            residual = x @ linear.T + cross * x * nxt + cubic * x**3 - y
            jac = np.broadcast_to(linear, x.shape[:-1] + (4, 4)).copy()
            jac[..., ids, ids] += cross * nxt + 3 * cubic * x**2
            jac[..., ids, after] += cross * x
            step = np.linalg.solve(jac, residual[..., None])[..., 0]
            x -= np.clip(step, -0.8, 0.8)
            if np.max(np.abs(step)) < 1e-10:
                break
        return x.sum(axis=-1)

    return stress

def _design(site, x, stress, dt, t):
    past, hold = stress(site['past_channel_raw']), stress(site['hold_channel_raw'])
    tail = np.cumsum(dt[::-1])[::-1] - dt
    features = np.column_stack((np.ones(len(x)), x))
    drift = features[:, None, :] * t[None, :, None]

    def matrices(taus):
        weights = -np.expm1(-dt[:, None] / taus) * np.exp(-tail[:, None] / taus)
        gap = hold[:, None] - past @ weights
        transient = gap[:, None, :] * -np.expm1(-t[:, None] / taus)
        return np.concatenate((transient, drift), axis=2), np.column_stack((gap / taus, features))

    return matrices


def _fit(matrices, strain, rate):
    target = np.r_[strain.ravel(), rate]
    n = strain.size

    def taus_of(p):
        first = np.exp(p[0])
        return np.array([first, first + (25.0 - first) * p[1]])

    def design(p):
        s, r = matrices(taus_of(p))
        return np.vstack((s.reshape(-1, 7), r))

    def mad(v):
        return max(1e-9, 1.4826 * np.median(np.abs(v - np.median(v))))

    def scales(residual):
        return np.r_[np.full(n, mad(residual[:n])), np.full(len(rate), mad(residual[n:]))]

    bounds = (np.r_[np.log(0.050001), 1e-6, np.full(7, -np.inf)],
              np.r_[np.log(24.999), 0.999999, np.full(7, np.inf)])
    initial = []
    for first, second in ((0.15, 0.9), (0.35, 2), (0.7, 5), (1.5, 10), (3, 18)):
        p = np.array([np.log(first), (second - first) / (25 - first)])
        a = design(p)
        beta = np.linalg.lstsq(a, target, rcond=None)[0]
        residual = a @ beta - target
        initial.append((np.r_[p, beta], residual))
    scale = scales(min(initial, key=lambda pair: np.median(np.abs(pair[1])))[1])

    def fit(p, scale, loss):
        return least_squares(lambda q: (design(q) @ q[2:] - target) / scale,
                             p, bounds=bounds, loss=loss, f_scale=2.0,
                             x_scale='jac', max_nfev=180)

    pilot = min((fit(p, scale, 'soft_l1') for p, _ in initial), key=lambda opt: opt.cost)
    scale = scales(pilot.fun * scale)
    solution = fit(pilot.x, scale, 'cauchy').x
    return taus_of(solution), solution[2:]


def _predict_chamber(site, x, stress, t, taus, theta):
    # The latest apparatus trace matches the archived reset regime, with a
    # logarithmic creep term with unit slope at the start of the hold.
    hold = stress(site['hold_channel_raw'])
    drift = theta[2] + x @ theta[3:]
    relaxation = -np.expm1(-t[:, None] / taus) @ theta[:2]
    strain = (site['strain_0'][:, None] + hold[:, None] * relaxation
              + drift[:, None] * np.log1p(t))
    rate = hold * np.sum(theta[:2] / taus) + drift
    return strain, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    taus, theta = _fit(_design(train, x_tr, stress, dt, t),
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _predict_chamber(test, x_te, stress, t, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
