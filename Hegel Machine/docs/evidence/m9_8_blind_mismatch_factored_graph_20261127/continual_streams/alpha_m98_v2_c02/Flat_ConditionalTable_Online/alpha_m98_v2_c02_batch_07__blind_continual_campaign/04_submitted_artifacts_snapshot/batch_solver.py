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
    """Fit each forward bridge channel, then invert the joint bridge map."""
    raw, loads = cal['bridge_raw'], cal['true_channel_loads']
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4

    def basis(x, channel):
        u = x[:, channel]
        return np.column_stack((np.ones(len(x)), u, x[:, nxt[channel]],
                                x[:, prev[channel]], u*x[:, nxt[channel]], u**3))

    designs = [basis(loads, channel) for channel in j]
    magnitude = np.max(np.abs(raw), axis=1)
    good = magnitude < 4*np.median(magnitude)
    for _ in range(3):
        b = np.column_stack([np.linalg.lstsq(a[good], raw[good, channel], rcond=None)[0]
                             for channel, a in enumerate(designs)])
        residual = np.max(np.abs(np.column_stack([a @ b[:, channel]
                                                  for channel, a in enumerate(designs)]) - raw), axis=1)
        good = residual <= 12*max(np.median(residual), 1e-8)

    linear = np.zeros((4, 4))
    linear[j, j], linear[nxt, j], linear[prev, j] = b[1:4]
    linear_inv = np.linalg.inv(linear)

    def stress(bridge):
        y = bridge.reshape(-1, 4)
        x = (y - b[0]) @ linear_inv
        for _ in range(12):
            neighbor = x[:, nxt]
            err = b[0] + x @ linear + b[4]*x*neighbor + b[5]*x**3 - y
            if np.max(np.abs(err)) < 1e-11:
                break
            jac = np.broadcast_to(linear.T, (len(x), 4, 4)).copy()
            jac[:, j, j] += b[4]*neighbor + 3*b[5]*x**2
            jac[:, j, nxt] += b[4]*x
            x -= np.linalg.solve(jac, err[..., None])[..., 0]
        return x.sum(axis=1).reshape(bridge.shape[:-1])

    return stress


def _design(site, x, stress, dt, t, *, chamber=False):
    past, hold = stress(site['past_channel_raw']), stress(site['hold_channel_raw'])
    tail = np.cumsum(dt[::-1])[::-1] - dt
    features = np.column_stack((np.ones(len(x)), x))
    clock = -np.expm1(-t) if chamber else t
    drift = features[:, None, :] * clock[None, :, None]

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
    taus, theta = _fit(_design(train, x_tr, stress, dt, t),
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Chamber retains the benchtop memory state. Its drift clock saturates
    # as 1 - exp(-t) and has the same unit derivative at hold onset.
    strain, rate = _design(test, x_te, stress, dt, t, chamber=True)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
