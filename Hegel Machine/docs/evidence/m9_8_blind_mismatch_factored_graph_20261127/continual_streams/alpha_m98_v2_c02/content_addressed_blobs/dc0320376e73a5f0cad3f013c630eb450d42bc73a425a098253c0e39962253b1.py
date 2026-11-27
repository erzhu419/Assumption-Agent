"""Fit the shared relaxation spectrum and predict blind chamber coupons."""

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


def _stress_calibration(cal):
    load = cal['true_channel_loads']
    raw = cal['bridge_raw']
    nxt = np.array([1, 2, 3, 0])
    cols = np.arange(4)
    base = np.column_stack((np.ones(len(load)), load))
    cubic = load**3
    cross = load * load[:, nxt]

    def fit(mask):
        return np.array([np.linalg.lstsq(
            np.column_stack((base, cubic[:, j], cross[:, j]))[mask],
            raw[mask, j], rcond=None)[0] for j in cols]).T

    # The mixed-load measurements identify all terms independently of the axis sweeps.
    coef = fit(np.count_nonzero(load, axis=1) > 1)
    err = np.linalg.norm(base @ coef[:5] + cubic * coef[5] + cross * coef[6] - raw, axis=1)
    median = np.median(err)
    mad = 1.4826 * np.median(np.abs(err - median))
    clean = err <= median + 10 * max(mad, 1e-8)
    coef = fit(clean)
    bias, linear, cubic_gain, cross_gain = coef[0], coef[1:5], coef[5], coef[6]

    def channel_loads(bridge):
        shape = bridge.shape
        y = bridge.reshape(-1, 4) - bias
        x = np.linalg.solve(linear.T, y.T).T
        for _ in range(15):
            z = x[:, nxt]
            r = x @ linear + x**3 * cubic_gain + x * z * cross_gain - y
            if np.max(np.abs(r)) < 1e-12:
                break
            jac = np.broadcast_to(linear.T, (len(x), 4, 4)).copy()
            jac[:, cols, cols] += 3 * x**2 * cubic_gain + z * cross_gain
            jac[:, cols, nxt] += x * cross_gain
            x -= np.linalg.solve(jac, r[..., None])[..., 0]
        return x.reshape(shape)

    def stress(bridge):
        return channel_loads(bridge).sum(axis=-1)

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


def _chamber(matrices, taus, theta, times, duration=1.0):
    strain, rate = matrices(taus)
    drift = rate[:, 2:] @ theta[2:]
    # The prefix apparatus trace shares the negative boundary-telemetry
    # signature of the archived gated-drift chamber, not the ungated lineage.
    integrated_gate = -duration * np.expm1(-times / duration)
    return strain[:, :, :2] @ theta[:2] + drift[:, None] * integrated_gate, rate @ theta


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    fit_design = _design(train, x_train, stress, dt, times)
    taus, theta = _fit(fit_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _chamber(_design(test, x_test, stress, dt, times), taus, theta, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
