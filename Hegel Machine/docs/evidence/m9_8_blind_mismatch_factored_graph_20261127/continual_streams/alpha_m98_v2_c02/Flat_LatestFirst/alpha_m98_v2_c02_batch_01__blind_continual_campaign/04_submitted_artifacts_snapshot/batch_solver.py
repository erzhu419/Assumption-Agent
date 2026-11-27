"""Fit the shared relaxation spectrum and predict chamber coupons."""

import numpy as np
from scipy.optimize import least_squares


def _reference_standardize(site):
    x = np.array(site['coupon_features_raw'], dtype=float, copy=True)
    lab = site['lab_id']
    ref = site['ref_coupons']
    for group in np.unique(lab):
        mask = lab == group
        controls = ref[mask]
        offset = np.nanmedian(np.nanmean(controls, axis=0))
        scale = np.nanmedian(np.nanstd(controls, axis=0))
        x[mask] = (x[mask] - offset) / (scale if scale > 0 else 1.0)
    return x


def _standardize(train, test):
    site = {k: np.concatenate((train[k], test[k])) for k in
            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')}
    x = _reference_standardize(site)
    _, inverse, counts = np.unique(np.column_stack((site['lab_id'], site['stratum'])),
                                   axis=0, return_inverse=True, return_counts=True)
    weight = (1.0 / counts[inverse]).reshape(-1, 1)
    observed = np.isfinite(x)
    fill = (np.sum(weight * np.where(observed, x, 0), axis=0)
            / np.sum(weight * observed, axis=0))
    x = np.where(observed, x, fill)
    return x[:len(train['lab_id'])]


def _chamber_features(site):
    x = _reference_standardize(site)
    return np.where(np.isnan(x), np.nanmean(x, axis=0), x)


def _stress_calibration(cal):
    raw = cal['bridge_raw']
    coefficients = np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))),
                                    cal['true_channel_loads'], rcond=None)[0].sum(axis=1)
    return lambda bridge: bridge @ coefficients[:4] + coefficients[4]


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


def _chamber_predict(site, stress, t, taus, theta):
    x = _chamber_features(site)
    drift = theta[2] + x @ theta[3:]
    # In this chamber the hold load resets the memory; the drift signal
    # enters both relaxation modes instead of accumulating linearly in time.
    amplitude = (stress(site['hold_channel_raw'])[:, None] * theta[:2]
                 + drift[:, None] * np.array([0.28, 2.09]))
    relaxation = -np.expm1(-t[:, None] / taus)
    return site['strain_0'][:, None] + amplitude @ relaxation.T, amplitude @ (1.0 / taus)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _chamber_predict(test, stress, t, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
