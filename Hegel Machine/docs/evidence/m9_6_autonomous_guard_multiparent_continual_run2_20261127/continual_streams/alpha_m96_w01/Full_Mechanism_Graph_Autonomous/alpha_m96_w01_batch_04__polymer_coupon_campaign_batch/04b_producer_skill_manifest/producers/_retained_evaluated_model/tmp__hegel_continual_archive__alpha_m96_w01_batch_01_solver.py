import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y, prior):
    i, j = np.triu_indices(len(x), 1)
    use = np.abs(x[j] - x[i]) > max(1e-8, .02 * np.ptp(x)) if len(x) else np.zeros(0, bool)
    if not use.any():
        a, b = prior
        if len(x):
            b = np.median(y - a*x)
        return a, b, False
    i, j = i[use], j[use]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    anchors = np.column_stack((slopes, y[i] - slopes*x[i]))
    anchors = np.vstack((anchors, prior))
    err = np.abs(x[None, :]*anchors[:, :1] + anchors[:, 1:] - y)
    # The load-cell checks resolve about 1e-4 in stress. Favor a line with
    # independent agreeing checks over a line fitted through an EMI spike.
    inliers = err <= .001
    count = inliers.sum(axis=1)
    tie = np.minimum(err, .001).sum(axis=1)
    best = np.lexsort((tie, -count))[0]
    keep = inliers[best]
    if keep.sum() < 2 or np.ptp(x[keep]) < 1e-8:
        a, b = prior
        return a, b, False
    if keep.sum() == 2 and len(x) > 2:
        close = np.argmin(np.abs(y - prior[0]*x - prior[1]))
        if np.abs(y[close] - prior[0]*x[close] - prior[1]) < .001:
            return prior[0], y[close] - prior[0]*x[close], False
    a, b = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
    return a, b, True


def _checks(c, lab):
    ids, x, y = (np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    m = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _calibrate(checks):
    prior = np.tile([1., 0.], (4, 1))
    if _ARCHIVE.is_file():
        with _ARCHIVE.open() as f:
            history = json.load(f)
        for c in history:
            for j in range(4):
                a, b, valid = _line(*_checks(c, j), prior[j])
                if valid:
                    prior[j] = a, b
    fitted = [_line(*_checks(checks, j), prior[j]) for j in range(4)]
    pars = np.array(fitted, float)
    return pars[:, 0], pars[:, 1], pars[:, 2].astype(bool)


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.empty_like(raw, dtype=float)
    for j in np.unique(lab):
        m = lab == j
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        shift = np.median(np.nanmean(r, axis=0)) if r.size else 0.
        scale = np.median(np.nanstd(r, axis=0)) if r.size else 1.
        x[m] = (raw[m] - shift) / (scale if scale > 0 else 1.)
    present = np.isfinite(x)
    cell_means = []
    for j, s in np.unique(np.column_stack((lab, strata)), axis=0):
        use = (lab == j) & (strata == s)
        count = present[use].sum(axis=0)
        cell_means.append(np.divide(np.nansum(x[use], axis=0), count,
                                    out=np.full(x.shape[1], np.nan), where=count > 0))
    cell_means = np.asarray(cell_means)
    count = np.isfinite(cell_means).sum(axis=0)
    fill = np.divide(np.nansum(cell_means, axis=0), count,
                     out=np.zeros(x.shape[1]), where=count > 0)
    return np.column_stack((np.ones(len(x)), np.where(present, x, fill))), present.all(axis=1)


def _design(tau, past, hold, initial, dt, times, x):
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = np.exp(-trailing[:, None]/tau) * -np.expm1(-dt[:, None]/tau)
    z = past @ kernel + initial*np.exp(-dt.sum()/tau)
    delta = hold[:, None] - z
    hold_basis = -np.expm1(-times[:, None]/tau)
    curves = np.concatenate((delta[:, None, :]*hold_basis,
                             x[:, None, :]*times[None, :, None]), axis=2)
    return curves, np.column_stack((delta/tau, x))


def _fit(past, hold, initial, dt, times, x, y, rate):
    cm, rm = np.isfinite(y), np.isfinite(rate)
    obs = np.r_[y[cm], rate[rm]]
    nc = cm.sum()

    def mat(tau):
        a, r = _design(tau, past, hold, initial, dt, times, x)
        return np.vstack((a[cm], r[rm]))

    def residual(p):
        return (np.einsum('ij,j->i', mat(np.exp(p[:2])), p[2:]) - obs) / scale

    starts = []
    grid = (.16, .33, .57, .95, 1.7, 3., 5., 9., 16.)
    for i, u in enumerate(grid):
        for v in grid[i+1:]:
            a = mat(np.array([u, v]))
            theta = np.linalg.lstsq(a, obs, rcond=None)[0]
            e = np.einsum('ij,j->i', a, theta) - obs
            w = 1 / np.sqrt(1 + (e / .035)**2)
            theta = np.linalg.lstsq(a*w[:, None], obs*w, rcond=None)[0]
            e = np.einsum('ij,j->i', a, theta) - obs
            starts.append((np.log1p((e/.035)**2).sum(), np.r_[np.log([u, v]), theta]))
    starts.sort(key=lambda pair: pair[0])
    bounds = (np.r_[np.log([.051, .051]), np.full(x.shape[1] + 2, -np.inf)],
              np.r_[np.log([24.9, 24.9]), np.full(x.shape[1] + 2, np.inf)])
    scale = .035
    fit = min((least_squares(residual, p, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=110)
               for _, p in starts[:3]), key=lambda f: f.cost)
    e = residual(fit.x)*scale

    def robust_scale(v):
        return max(.001, 2.5*1.4826*np.median(np.abs(v - np.median(v)))) if v.size else .035

    scale = np.r_[np.full(nc, robust_scale(e[:nc])), np.full(len(e)-nc, robust_scale(e[nc:]))]
    fit = least_squares(residual, fit.x, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=120)
    order = np.argsort(fit.x[:2])
    return np.exp(fit.x[:2][order]), np.r_[fit.x[2:4][order], fit.x[4:]]


def _refine_sparse(tr, checks, gain, bias, identified, tau, theta, dt, times, y):
    # Remove the feature-dependent linear drift before estimating the load-cell
    # line. The remaining time curvature comes from calibrated stress and memory.
    relax = -np.expm1(-times[:, None] / tau)
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = np.exp(-trailing[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    survival = np.exp(-dt.sum() / tau)
    modes = (relax - np.outer(times, (times @ relax) / (times @ times))) * theta[:2]
    for lab in np.flatnonzero(~identified):
        use = (tr['lab_id'] == lab) & np.isfinite(y).all(axis=1)
        if use.sum() < 4:
            continue
        curve = y[use]
        projected = curve - np.outer((curve @ times) / (times @ times), times)
        command = tr['stress_hold_cmd'][use, None] - tr['past_stress_cmd'][use] @ kernel
        memory = tr['initial_memory_states'][use] * survival
        a = (command @ modes.T).ravel()
        b = (survival @ modes.T)
        mat = np.column_stack((a, np.tile(b, use.sum())))
        obs = (projected + memory @ modes.T).ravel()
        # Scale the columns separately; a constant stress offset can have a
        # much smaller hereditary signature than a gain shift.
        norm = np.linalg.norm(mat, axis=0)
        if norm[0] == 0:
            continue
        if norm[1] > 1e-8 * norm[0]:
            bfit = least_squares(lambda v: (mat / norm) @ v - obs,
                                 np.array([gain[lab], bias[lab]]) * norm,
                                 loss='soft_l1', f_scale=.025).x / norm
        else:
            bfit = np.array([gain[lab], bias[lab]])
        x, z = _checks(checks, lab)
        if len(x):
            distance = np.abs(z - bfit[0] * x - bfit[1])
            k = distance.argmin()
            if distance[k] < .12:
                anchor_x, anchor_y = x[k], z[k]
                column = mat[:, 0] - anchor_x * mat[:, 1]
                target = obs - anchor_y * mat[:, 1]
                fitted = least_squares(lambda v: column * v[0] - target,
                                       [bfit[0]], loss='soft_l1', f_scale=.025).x[0]
                bfit = [fitted, anchor_y - anchor_x * fitted]
        if bfit[0] > 0:
            gain[lab], bias[lab] = bfit


def _allocation(d, strain):
    z = np.asarray(d['Z_nuisance'], float)
    pz = np.linalg.solve(z.T @ z + float(d['nuisance_ridge'])*np.eye(z.shape[1]), z.T)
    s = np.asarray(d['S_alloc_modes'], float) * (1 + .25*np.tanh(strain[:, -1] - d['test']['strain_0']))[:, None]
    y = np.asarray(d['y_alloc_obs'], float).ravel()
    s, y = s - z @ (pz @ s), y - z @ (pz @ y)
    ridge, cap, budget = (float(d[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    m = s.shape[1]
    h, c = s.T @ s + ridge*np.eye(m), s.T @ y
    if budget <= 0 or budget >= m*cap:
        w = np.full(m, 0. if budget <= 0 else cap)
    else:
        norm = max(np.max(np.abs(h)), np.max(np.abs(c)), 1.)
        fit = minimize(lambda w: (.5*w @ h @ w - c @ w)/norm, np.full(m, budget/m),
                       jac=lambda w: (h @ w - c)/norm, method='SLSQP',
                       bounds=[(0., cap)]*m,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum()-budget,
                                    'jac': lambda w: np.ones(m)},
                       options={'ftol': 1e-13, 'maxiter': 300})
        w = np.clip(fit.x, 0., cap)
        # Solve the equality-constrained system on the estimated active face.
        lo, hi = w <= 1e-8*cap, w >= (1.-1e-8)*cap
        free = ~(lo | hi)
        n = free.sum()
        if n:
            hf = h[np.ix_(free, free)]
            system = np.block([[hf, -np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
            rhs = np.r_[c[free] - cap*h[np.ix_(free, hi)].sum(axis=1), budget - cap*hi.sum()]
            sol = np.linalg.solve(system, rhs)
            if sol[:-1].min() >= -1e-10 and sol[:-1].max() <= cap + 1e-10:
                w[lo], w[hi], w[free] = 0., cap, np.clip(sol[:-1], 0., cap)
    res = s @ w - y
    p = .5*(res @ res + ridge*(w @ w))
    g = s.T @ res + ridge*w
    # A fractional budget has its tangent LP threshold at the marginal mode.
    nu = np.sort(g)[min(max(int(np.ceil(budget/cap))-1, 0), m-1)] if cap else 0.
    low, high = np.maximum(g-nu, 0.), np.maximum(nu-g, 0.)
    dual = min(p, p - g @ w + nu*budget - cap*high.sum())
    cert = dict(weights=w, nu_sum=float(nu), lam_lower=low, lam_upper=high,
                primal_objective=float(p), dual_lower_bound=float(dual))
    return w, cert


def solve_alpha_m96_w01_batch_01(dataset: dict) -> dict:
    d = dataset
    tr, te = d['train'], d['test']
    x, complete = _features(tr, te)
    n = len(tr['lab_id'])
    gain, bias, identified = _calibrate(d['calibration_load_cells'])
    dt, times = np.asarray(d['past_dt']), np.asarray(d['eval_times'])

    def inputs(split):
        lab = np.asarray(split['lab_id'], int)
        return (np.asarray(split['past_stress_cmd'])*gain[lab, None] + bias[lab, None],
                np.asarray(split['stress_hold_cmd'])*gain[lab] + bias[lab],
                np.asarray(split['initial_memory_states']))

    y = np.asarray(tr['strain_curves_obs']) - np.asarray(tr['strain_0'])[:, None]
    rate = np.asarray(tr['instant_rate_obs'])
    use = complete[:n] if complete[:n].sum() >= 20 else np.ones(n, bool)
    trusted = use & identified[np.asarray(tr['lab_id'], int)]

    def fit(rows):
        return _fit(*(a[rows] for a in inputs(tr)), dt, times, x[:n][rows],
                    y[rows], rate[rows])

    if not identified.all() and trusted.sum() >= 20:
        tau, theta = fit(trusted)
        _refine_sparse(tr, d['calibration_load_cells'], gain, bias,
                       identified, tau, theta, dt, times, y)
    tau, theta = fit(use)
    a, r = _design(tau, *inputs(te), dt, times, x[n:])
    strain = a @ theta + np.asarray(te['strain_0'])[:, None]
    # Missing features leave an unknown constant contribution to creep rate.
    # Observe that contribution at hold onset without altering the relaxation.
    rate_obs = np.asarray(te['instant_rate_obs'])
    correct = ~complete[n:] & np.isfinite(rate_obs)
    strain[correct] += (rate_obs[correct] - r[correct] @ theta)[:, None] * times
    w, cert = _allocation(d, strain)
    return dict(estimated_taus=tau, predicted_strain=strain, allocation_weights=w, kkt_certificate=cert)
