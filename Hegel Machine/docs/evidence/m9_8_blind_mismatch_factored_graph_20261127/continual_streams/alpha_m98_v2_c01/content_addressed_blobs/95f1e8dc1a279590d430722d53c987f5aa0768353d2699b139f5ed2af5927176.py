"""Calibrate the bridge, fit the benchtop spectrum, and predict chamber creep."""
import numpy as np
from scipy.optimize import least_squares


def _axial_load_map(calibration):
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    # The failed sweep has a common large offset across all four channels.
    magnitude = np.linalg.norm(raw - np.median(raw, axis=0), axis=1)
    good = magnitude < 4 * np.median(magnitude)
    loads, raw = loads[good], raw[good]
    # Each output has its own cubic and couples to its clockwise neighbor.
    coeff = np.empty((4, 6))
    for i in range(4):
        x, y = loads[:, i], loads[:, (i + 1) % 4]
        z = loads[:, (i - 1) % 4]
        design = np.column_stack((np.ones(len(x)), x, y, z, x*y, x**3))
        coeff[i] = np.linalg.lstsq(design, raw[:, i], rcond=None)[0]
    return coeff


def _decode_axial(raw, coeff):
    raw = np.asarray(raw, dtype=float)
    shape = raw.shape[:-1]
    raw = raw.reshape(-1, 4)
    bias, own, nxt, prev, cross, cubic = coeff.T
    j = np.arange(4)
    lin = np.diag(own)
    lin[j, (j+1) % 4] = nxt
    lin[j, (j-1) % 4] = prev
    x = np.linalg.solve(lin, (raw - bias).T).T
    jac = np.broadcast_to(lin, (len(x), 4, 4)).copy()
    for _ in range(30):
        y = np.roll(x, -1, axis=1)
        r = x @ lin.T + bias + cross*x*y + cubic*x**3 - raw
        if np.max(np.abs(r)) < 1e-11:
            break
        jac[:, j, j] = own + cross*y + 3*cubic*x*x
        jac[:, j, (j+1) % 4] = nxt + cross*x
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        # Cap the first Newton steps at one load unit.
        x -= step / np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True))
    return x.sum(axis=-1).reshape(shape)


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


def _specimen(group, features, bridge):
    past = _decode_axial(group['past_channel_raw'], bridge)
    hold = _decode_axial(group['hold_channel_raw'], bridge)
    return past, hold, np.column_stack((np.ones(len(hold)), features))

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


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    xtr, xte = _features(train, test)
    bridge = _axial_load_map(dataset['calibration_load_cells'])
    tr = _specimen(train, xtr, bridge)
    tau, theta = _fit(tr, dt, times, train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])
    te = _specimen(test, xte, bridge)
    curves, rates = _branches(te, dt, times, tau)
    drift = te[2] @ theta[2:]
    # The prefix apparatus trace matches the archived exponential-drift chamber regime.
    # Its initial drift rate is unchanged.
    drift_time = -np.expm1(-times)
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curves @ theta[:2] + drift[:, None] * drift_time,
            'initial_rate_pred': rates @ theta[:2] + drift}
