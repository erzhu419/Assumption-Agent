"""Fit the load bridge and intrinsic relaxation; predict chamber strain and initial rate."""
import numpy as np
from scipy.optimize import least_squares


def fit_bridge(cal):
    a = np.asarray(cal['true_channel_loads'], float)
    b = np.asarray(cal['bridge_raw'], float)
    nxt = [1, 2, 3, 0]
    prv = [3, 0, 1, 2]
    # Gross bridge errors are large even when their sum cancels.
    center = np.median(b, axis=0)
    spread = np.median(abs(b-center), axis=0)
    keep = np.max(abs(b-center)/np.maximum(spread, .1), axis=1) < 12
    co = np.array([np.linalg.lstsq(np.column_stack((np.ones(len(a)), a[:,j], a[:,nxt[j]], a[:,prv[j]], a[:,j]*a[:,nxt[j]], a[:,j]**3))[keep], b[keep,j], rcond=None)[0] for j in range(4)])
    off, lin, cross_next, cross_prev, pair, cubic = co.T
    linear = np.diag(lin)
    linear[np.arange(4),nxt] = cross_next
    linear[np.arange(4),prv] = cross_prev

    def invert(raw):
        y = np.asarray(raw, float)-off
        q = np.linalg.solve(linear, y.reshape(-1,4).T).T
        y = y.reshape(-1,4)
        for _ in range(24):
            f = q@linear.T + pair*q*q[:,nxt] + cubic*q**3-y
            if np.max(abs(f)) < 1e-11:
                break
            jac = np.broadcast_to(linear,(len(q),4,4)).copy()
            j = np.arange(4)
            jac[:,j,j] += pair*q[:,nxt]+3*cubic*q*q
            jac[:,j,nxt] += pair*q
            step = np.linalg.solve(jac, f[...,None])[...,0]
            q -= step/np.maximum(1.,np.max(abs(step),axis=1,keepdims=True))
        return q.reshape(raw.shape)
    return invert, keep


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


def _specimen(group, x, invert):
    return (invert(group['past_channel_raw']).sum(axis=-1),
            invert(group['hold_channel_raw']).sum(axis=-1),
            np.column_stack((np.ones(len(x)), x)))


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
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    invert, _ = fit_bridge(dataset['calibration_load_cells'])
    xtr, xte = _features(train, test)
    taus, weights = _fit(_specimen(train, xtr, invert), dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    specimen = _specimen(test, xte, invert)
    curves, rates = _branches(specimen, dt, times, taus)
    drift = specimen[2] @ weights[2:]
    # The chamber preserves the loading history, but its drift rate decays as exp(-t).
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'predicted_strain': (test['strain_0'][:, None] + curves @ weights[:2]
                                 + drift[:, None] * (-np.expm1(-times))),
            'initial_rate_pred': rates @ weights[:2] + drift}
