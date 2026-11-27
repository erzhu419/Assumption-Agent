"""Calibrate the load bridge and fit a shared two-branch relaxation spectrum."""
import numpy as np
from scipy.optimize import least_squares


def _axial_load_map(calibration):
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    # Reject gross common-offset faults in otherwise small calibration sweeps.
    clean = np.max(np.abs(raw - np.median(raw, axis=0)), axis=1) < 8.0
    x, y = loads[clean], raw[clean]
    coeff = np.empty((4, 7))
    for j in range(4):
        u, v = x[:, j], x[:, (j + 1) % 4]
        design = np.column_stack((np.ones(len(x)), x, u * v, u**3))
        coeff[j] = np.linalg.lstsq(design, y[:, j], rcond=None)[0]
    return coeff


def _channel_loads(raw, coeff):
    linear = coeff[:, 1:5]
    pair, cube = coeff[:, 5], coeff[:, 6]
    loads = (raw - coeff[:, 0]) @ np.linalg.inv(linear.T)
    j = np.arange(4)
    successor = (j + 1) % 4
    for _ in range(18):
        neighbor = loads[..., successor]
        error = (loads @ linear.T + coeff[:, 0] + pair * loads * neighbor
                 + cube * loads**3 - raw)
        jac = np.broadcast_to(linear, loads.shape[:-1] + (4, 4)).copy()
        jac[..., j, j] += pair * neighbor + 3 * cube * loads**2
        jac[..., j, successor] += pair * loads
        step = np.linalg.solve(jac, error[..., None])[..., 0]
        loads -= np.clip(step, -0.7, 0.7)
        if np.max(np.abs(step), initial=0.0) < 1e-10:
            break
    return loads


def _features(train, test):
    groups = (train, test)
    raw, ref, lab, strata = (np.concatenate([d[k] for d in groups], axis=0) for k in
                            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty(raw.shape, dtype=float)
    for key in np.unique(lab):
        rows = lab == key
        panel = ref[rows]
        center = np.median(np.mean(panel, axis=0))
        scale = np.median(np.std(panel, axis=0, ddof=0))
        x[rows] = (raw[rows] - center) / (scale if scale > 0 else 1.)
    _, cell, counts = np.unique(np.column_stack((lab, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    present = np.isfinite(x)
    denom = np.sum(w * present, axis=0)
    center = np.sum(w * np.where(present, x, 0.), axis=0) / np.where(denom > 0, denom, 1.)
    x = np.where(present, x, center)
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _specimen(group, x, bridge):
    past = _channel_loads(group['past_channel_raw'], bridge).sum(axis=-1)
    hold = _channel_loads(group['hold_channel_raw'], bridge).sum(axis=-1)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _branches(specimen, dt, times, taus, derivatives=False):
    past, hold, _ = specimen
    u = dt[:, None] / taus
    decay = np.exp(-u)
    increment = -np.expm1(-u)
    state = np.zeros((len(hold), 2))
    dstate = np.zeros_like(state)
    for j in range(len(dt)):
        stress = past[:, j, None]
        if derivatives:
            dstate = decay[j] * (dstate + u[j] * (state - stress))
        state = decay[j] * state + increment[j] * stress
    gap = hold[:, None] - state
    v = times[:, None] / taus
    relax = -np.expm1(-v)
    curves = gap[:, None, :] * relax
    rates = gap / taus
    if not derivatives:
        return curves, rates
    dcurves = -dstate[:, None, :] * relax - gap[:, None, :] * v * np.exp(-v)
    drates = -dstate / taus - rates
    return curves, rates, dcurves, drates


def _fit(specimen, dt, times, curves_obs, rate_obs):
    n, k = curves_obs.shape
    drift = specimen[2]
    fixed = np.vstack(((times[None, :, None] * drift[:, None, :]).reshape(n*k, 5), drift))
    obs = np.concatenate((curves_obs.ravel(), rate_obs))
    # Curve and rate channels have separate noise scales. Rates count as
    # independent observations rather than as a coupon-level penalty.
    group = np.arange(len(obs)) >= n*k
    floor = 1e-6 * max(1., np.median(np.abs(obs)))

    def design(tau):
        c, r = _branches(specimen, dt, times, tau)
        return np.column_stack((np.vstack((c.reshape(n*k, 2), r)), fixed))

    def mad(err):
        return max(floor, 1.4826 * np.median(np.abs(err - np.median(err))))

    grid = np.geomspace(.065, 23., 12)
    seeds = []
    for j, a in enumerate(grid[:-1]):
        for b in grid[j+1:]:
            mat = design((a, b))
            coef = np.linalg.lstsq(mat, obs, rcond=None)[0]
            err = mat @ coef - obs
            seeds.append((mad(err[~group]) + .35 * mad(err[group]), a, b, coef))
    seeds.sort(key=lambda s: s[0])
    _, a, b, coef = seeds[0]
    e = design((a, b)) @ coef - obs
    scales = np.where(group, mad(e[group]), mad(e[~group]))
    # Equal bounds on branch lifetimes preserve label symmetry; sort only at output.
    bounds = (np.r_[np.log([.05000001]*2), [-np.inf]*7],
              np.r_[np.log([24.999999]*2), [np.inf]*7])

    def fun(p):
        return (design(np.exp(p[:2])) @ p[2:] - obs) / scales

    def jac(p):
        c, r, dc, dr = _branches(specimen, dt, times, np.exp(p[:2]), derivatives=True)
        var = np.vstack((c.reshape(n*k, 2), r))
        dvar = np.vstack((dc.reshape(n*k, 2), dr)) * p[2:4]
        return np.column_stack((dvar, var, fixed)) / scales[:, None]

    fits = [least_squares(fun, np.r_[np.log([a,b]), coef], jac=jac, bounds=bounds,
                          loss='soft_l1', x_scale='jac', max_nfev=80)
            for _, a, b, coef in seeds[:4]]
    best = min(fits, key=lambda fit: fit.cost)
    for _ in range(2):
        e = design(np.exp(best.x[:2])) @ best.x[2:] - obs
        scales = np.where(group, mad(e[group]), mad(e[~group]))
        best = least_squares(fun, best.x, jac=jac, bounds=bounds,
                             loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    tau = np.exp(best.x[:2]); theta = best.x[2:].copy()
    order = np.argsort(tau)
    tau = tau[order]; theta[:2] = theta[:2][order]
    return tau, theta



def _chamber_increment(hold, features, times, tau, theta):
    """The chamber resets branch memory and replaces linear creep with log creep."""
    drift = features @ theta[2:]
    amplitude = hold[:, None] * theta[:2]
    strain = amplitude @ (-np.expm1(-times[:, None] / tau)).T
    strain += drift[:, None] * np.log1p(times)
    rate = amplitude @ (1.0 / tau) + drift
    return strain, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _axial_load_map(dataset['calibration_load_cells'])
    x_train, x_test = _features(train, test)
    source = _specimen(train, x_train, bridge)
    tau, theta = _fit(source, dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])
    _, hold, features = _specimen(test, x_test, bridge)
    increment, rate = _chamber_increment(hold, features, times, tau, theta)
    return dict(estimated_taus=tau, hereditary_weights=theta,
                predicted_strain=test['strain_0'][:, None] + increment, initial_rate_pred=rate)
