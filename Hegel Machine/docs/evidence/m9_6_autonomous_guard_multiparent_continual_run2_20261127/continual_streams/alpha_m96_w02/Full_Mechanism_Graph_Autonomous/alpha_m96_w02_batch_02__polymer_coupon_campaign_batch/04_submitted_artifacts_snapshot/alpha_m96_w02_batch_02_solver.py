"""Preloaded hereditary creep and certified compliance allocation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


def _stress_calibration(checks):
    lab = np.asarray(checks['lab_id'])
    cmd = np.asarray(checks['cmd_stress'], float)
    true = np.asarray(checks['true_stress'], float)
    history_path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    params = np.tile([1., 0.], (4, 1))
    for c in range(4):
        past = []
        for sweep in history:
            use = np.asarray(sweep['lab_id']) == c
            x = np.asarray(sweep['cmd_stress'])[use]
            y = np.asarray(sweep['true_stress'])[use]
            if len(x) > 1 and np.ptp(x) > 0.1:
                past.append(np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0])
        prior = past[-1] if past else params[c].copy()
        params[c] = prior
        use = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[use], true[use]
        if not len(x):
            continue
        i, j = np.triu_indices(len(x), 1)
        distinct = abs(x[i] - x[j]) > 0.05
        i, j = i[distinct], j[distinct]
        slopes = np.r_[prior[0], (y[i] - y[j]) / (x[i] - x[j])]
        offsets = np.median(y[None, :] - slopes[:, None] * x[None, :], axis=1)
        r = abs(y[None, :] - slopes[:, None] * x[None, :] - offsets[:, None])
        best = np.argmin(np.median(r, axis=1))
        cutoff = max(7e-4, 4.5 * 1.4826 * np.median(r[best]))
        inliers = r[best] <= cutoff
        if np.ptp(x[inliers]) > 0.1:
            xx, yy = x[inliers], y[inliers]
            params[c] = np.linalg.lstsq(np.column_stack((xx, np.ones(len(xx)))), yy, rcond=None)[0]
        else:
            params[c, 1] = np.median(y[inliers] - prior[0] * x[inliers])
    return params


def _align(train, test):
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    stratum = np.concatenate((train['stratum'], test['stratum']))
    x = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0))
        x[use] = (x[use] - offset) / gain
    # Give each observed lab/stratum cell equal row mass before excluding missing values.
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    valid = np.isfinite(x)
    mass = np.sum(w * valid, axis=0)
    fill = np.divide(np.sum(w * np.where(valid, x, 0.), axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0.)
    return np.where(valid, x, fill)


def _inputs(data, x, calibration):
    gain, bias = calibration[np.asarray(data['lab_id'])].T
    past = gain[:, None] * data['past_stress_cmd'] + bias[:, None]
    hold = gain * data['stress_hold_cmd'] + bias
    return past, hold, np.column_stack((np.ones(len(x)), x)), np.asarray(data["initial_memory_states"], float)


class _Design:
    def __init__(self, dt, t, inputs):
        self.dt, self.t = np.asarray(dt), np.asarray(t)
        self.past, self.hold, self.x, self.initial = inputs
        self.base = np.empty((len(self.x), len(t) + 1, 7))
        self.base[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:] = self.x

    def make(self, taus, derivative=False):
        a = self.base.copy()
        if derivative:
            da = np.empty(a.shape[:2] + (2,))
        for m, tau in enumerate(taus):
            z = self.initial[:, m].copy()
            dz = np.zeros_like(z)
            e = np.exp(-self.dt / tau)
            f = -np.expm1(-self.dt / tau)
            de = e * self.dt / tau**2
            for j in range(len(e)):
                if derivative:
                    dz = e[j] * dz + de[j] * (z - self.past[:, j])
                z = e[j] * z + f[j] * self.past[:, j]
            gap = self.hold - z
            h = -np.expm1(-self.t / tau)
            a[:, :-1, m] = gap[:, None] * h
            a[:, -1, m] = gap / tau
            if derivative:
                dh = -np.exp(-self.t / tau) * self.t / tau**2
                da[:, :-1, m] = gap[:, None] * dh - dz[:, None] * h
                da[:, -1, m] = -dz / tau - gap / tau**2
        return (a, da) if derivative else a


def _fit(design, curves, initial_rate):
    y = np.column_stack((curves, initial_rate))
    def scales(r):
        def mad(a):
            return 1.4826 * np.median(abs(a - np.median(a)))
        s = max(1e-7, mad(r[:, :-1]))
        return np.r_[np.full(y.shape[1] - 1, s), max(1e-7, mad(r[:, -1]))]

    def linear_start(a):
        a = a.reshape(-1, 7)
        yflat = y.ravel()
        b = np.linalg.lstsq(a, yflat, rcond=None)[0]
        for _ in range(3):
            r = (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)
            s = 2.5 * scales(r)
            w = (1. / np.hypot(s, r)).ravel()
            b = np.linalg.lstsq(a * w[:, None], yflat * w, rcond=None)[0]
        return b, (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)

    seeds = []
    grid = (.22, .38, .68, 1.20, 2.15, 3.8, 6.5, 7.9)
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            b, r = linear_start(design.make((t1, t2)))
            seeds.append((np.r_[t1, t2, b], r))
    pilot = min(seeds, key=lambda sr: np.median(abs(sr[1][:, :-1])))
    scale = 2.5 * scales(pilot[1])
    bounds = (np.r_[.20, .20, np.full(7, -np.inf)], np.r_[8., 8., np.full(7, np.inf)])

    def score(r):
        return np.sum(np.log1p((r / scale)**2))

    def optimize(p):
        def residual(q):
            a = design.make(q[:2])
            return ((np.einsum('ntj,j->nt', a, q[2:]) - y) / scale).ravel()

        def jac(q):
            a, da = design.make(q[:2], derivative=True)
            j = np.concatenate((da * q[2:4], a), axis=2)
            return (j / scale[None, :, None]).reshape(-1, 9)

        return least_squares(residual, p, jac=jac, bounds=bounds, loss='cauchy',
                             tr_solver='exact', x_scale='jac', max_nfev=120,
                             ftol=2e-9, xtol=2e-9, gtol=2e-7)

    seeds.sort(key=lambda sr: score(sr[1]))
    fitted = min((optimize(p) for p, _ in seeds[:4]), key=lambda result: result.cost)
    scale = 2.5 * scales(fitted.fun.reshape(y.shape) * scale)
    p = optimize(fitted.x).x
    return p[:2], p[2:]


def _allocation(dataset, strain):
    z = np.asarray(dataset['Z_nuisance'], float)
    pz = np.linalg.solve(z.T @ z + dataset['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    modulation = 1. + .25 * np.tanh(strain[:, -1] - dataset['test']['strain_0'])
    s = np.asarray(dataset['S_alloc_modes']) * modulation[:, None]
    y = np.asarray(dataset['y_alloc_obs']).ravel()
    a = s - z @ (pz @ s)
    b = y - z @ (pz @ y)
    ridge = float(dataset['source_ridge'])
    u, budget = float(dataset['u_box']), float(dataset['total_budget'])
    n = a.shape[1]
    q = a.T @ a + ridge * np.eye(n)
    linear = a.T @ b

    def value(w):
        r = a @ w - b
        return .5 * (r @ r + ridge * (w @ w))

    def grad(w):
        return q @ w - linear

    w = minimize(value, np.full(n, budget / n), jac=grad, method='SLSQP',
                 bounds=[(0., u)] * n,
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget,
                               'jac': lambda w: np.ones(n)}],
                 options={'ftol': 1e-13, 'maxiter': 300}).x
    # Polish the identified faces by solving the equality-constrained KKT system.
    lo, hi = w < 1e-8, w > u - 1e-8
    f = ~(lo | hi)
    if np.any(f):
        ones = np.ones((f.sum(), 1))
        kkt = np.block([[q[np.ix_(f, f)], ones], [ones.T, np.zeros((1, 1))]])
        sol = np.linalg.solve(kkt, np.r_[linear[f] - q[np.ix_(f, hi)].sum(axis=1) * u, budget - u * hi.sum()])
        if np.all((sol[:-1] >= 0.) & (sol[:-1] <= u)):
            w[lo], w[hi], w[f] = 0., u, sol[:-1]
        nu = float(sol[-1])
    else:
        nu = float(np.clip(0., -np.min(grad(w)[lo], initial=np.inf), -np.max(grad(w)[hi], initial=-np.inf)))
    g = grad(w) + nu
    lower = np.where(lo, np.maximum(g, 0.), 0.)
    upper = np.where(hi, np.maximum(-g, 0.), 0.)
    # The certificate uses the equality term nu_sum * (budget - sum(w)).
    nu_sum = -nu
    primal = value(w)
    grad_w = a.T @ (a @ w - b) + ridge * w
    dual = primal - grad_w @ w + nu_sum * budget
    dual += u * np.minimum(grad_w - nu_sum, 0.).sum()
    return w, {'weights': w, 'nu_sum': nu_sum, 'lam_lower': lower, 'lam_upper': upper,
               'primal_objective': float(primal), 'dual_lower_bound': float(min(primal, dual))}



def _rate_adjustment(train, test, train_design, test_matrix, taus, theta):
    """Condition the missing-feature drift on the initial-rate measurement."""
    train_missing = ~np.isfinite(train['coupon_features_raw'])
    test_missing = ~np.isfinite(test['coupon_features_raw'])
    delta = np.zeros(len(test_missing))
    need = test_missing.any(axis=1) & np.isfinite(test['instant_rate_obs'])
    if not np.any(need):
        return delta

    # The hereditary modes already contribute to the initial derivative. Only
    # the unexplained part can update the coupon's time-linear drift.
    complete = ~train_missing.any(axis=1)
    rate_resid = np.asarray(train['instant_rate_obs']) - train_design.make(taus)[:, -1] @ theta
    noise_sample = rate_resid[complete & np.isfinite(rate_resid)]
    noise = 1.4826 * np.median(abs(noise_sample - np.median(noise_sample))) if noise_sample.size else 0.
    aligned = train_design.x[:, 1:]
    spread = np.array([np.std(aligned[~train_missing[:, j], j])
                       if np.any(~train_missing[:, j]) else 1.
                       for j in range(aligned.shape[1])])
    variance = test_missing[need] @ (spread * theta[3:])**2
    denom = variance + noise**2
    gain = np.divide(variance, denom, out=np.zeros_like(variance), where=denom > 0.)
    delta[need] = gain * (np.asarray(test['instant_rate_obs'])[need] - test_matrix[need, -1] @ theta)
    return delta


def solve_alpha_m96_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    cal = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    design = _Design(dt, t, _inputs(train, x[:n], cal))
    curves = train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None]
    taus, theta = _fit(design, curves, train['instant_rate_obs'])
    test_design = _Design(dt, t, _inputs(test, x[n:], cal))
    matrix = test_design.make(taus)
    delta = _rate_adjustment(train, test, design, matrix, taus, theta)
    strain = matrix[:, :-1] @ theta + np.asarray(test['strain_0'])[:, None] + delta[:, None] * t
    w, cert = _allocation(dataset, strain)
    return {'estimated_taus': taus, 'predicted_strain': strain,
            'allocation_weights': w, 'kkt_certificate': cert}
