"""Fit benchtop relaxation and predict chamber strain and initial rate."""

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
    raw, loads = cal['bridge_raw'], cal['true_channel_loads']
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]

    def basis(x):
        return np.column_stack((np.ones(len(x)), x,
                                *(x[:, i]*x[:, j] for i, j in pairs), x**3))

    a = basis(loads)
    magnitude = np.max(np.abs(raw), axis=1)
    good = magnitude < 4*np.median(magnitude)
    b = np.linalg.lstsq(a[good], raw[good], rcond=None)[0]
    # Use forward residuals to reject calibration points displaced off the bridge law.
    errors = np.max(np.abs(a @ b - raw), axis=1)
    typical = np.median(errors)
    good = errors <= 12*max(typical, 1e-8)
    b = np.linalg.lstsq(a[good], raw[good], rcond=None)[0]
    linear_inv = np.linalg.inv(b[1:5])

    def stress(bridge):
        y = bridge.reshape(-1, 4)
        x = (y - b[0]) @ linear_inv
        for _ in range(12):
            err = basis(x) @ b - y
            if np.max(np.abs(err)) < 1e-11:
                break
            jac = np.broadcast_to(b[1:5], (len(x), 4, 4)).copy()
            for row, (i, j) in enumerate(pairs, start=5):
                jac[:, i] += x[:, j, None]*b[row]
                jac[:, j] += x[:, i, None]*b[row]
            jac += 3*x[:, :, None]**2*b[15:19]
            step = np.linalg.solve(jac.transpose(0, 2, 1), err[..., None])[..., 0]
            x -= step
        return x.sum(axis=1).reshape(bridge.shape[:-1])

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


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Chamber entry resets both memory branches; logarithmic creep has unit initial slope.
    hold = stress(test['hold_channel_raw'])
    drift = theta[2] + x_te @ theta[3:]
    relaxation = -np.expm1(-t[:, None] / taus) @ theta[:2]
    strain = test['strain_0'][:, None] + hold[:, None] * relaxation + drift[:, None] * np.log1p(t)
    rate = hold * np.sum(theta[:2] / taus) + drift
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
