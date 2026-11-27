"""Calibrated sub-yield creep fit for the multi-wing coupon stream."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


def _calibration_prior():
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    if not path.is_file():
        return np.tile([1., 0.], (4, 1))
    history = json.loads(path.read_text())
    params = []
    for c in range(4):
        estimates = []
        for sweep in history:
            mask = np.asarray(sweep['lab_id']) == c
            x = np.asarray(sweep['cmd_stress'])[mask]
            y = np.asarray(sweep['true_stress'])[mask]
            estimates.append(np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0])
        params.append(np.median(estimates, axis=0))
    return np.asarray(params)


def _stress_calibration(checks):
    lab = np.asarray(checks['lab_id'])
    cmd = np.asarray(checks['cmd_stress'], float)
    true = np.asarray(checks['true_stress'], float)
    params = _calibration_prior()
    for c, (gain, bias) in enumerate(params.copy()):
        use = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[use], true[use]
        if not len(x):
            continue
        i, j = np.triu_indices(len(x), 1)
        pairs = abs(x[i] - x[j]) >= .1
        i, j = i[pairs], j[pairs]
        slopes = (y[i] - y[j]) / (x[i] - x[j])
        # Try both in-batch lines and the historical slope anchored at each spot check.
        lines = np.column_stack((np.r_[slopes, np.full(len(x) + 1, gain)],
                                 np.r_[y[i] - slopes * x[i], y - gain * x, bias]))
        A = np.column_stack((x, np.ones(len(x))))
        errors = abs(A @ lines.T - y[:, None])
        inliers = errors <= 7e-4
        count = inliers.sum(axis=0)
        # With comparable support, prefer the line nearest the preceding clean sweeps.
        departure = np.linalg.norm(lines - [gain, bias], axis=1)
        best = np.lexsort((departure, -count))[0]
        use = inliers[:, best]
        if count[best] >= 3 and np.ptp(x[use]) >= .1:
            params[c] = np.linalg.lstsq(A[use], y[use], rcond=None)[0]
        elif count[best] == 2 and len(x) == 2 and np.ptp(x) >= .1 and departure[best] < .03:
            params[c] = np.linalg.solve(A, y)
        elif count[best] >= 2 or (len(x) == 1 and departure[best] < .01):
            params[c, 1] = np.median(y[use] - gain * x[use])
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
    return past, hold, np.column_stack((np.ones(len(x)), x)), np.asarray(data['initial_memory_states'], float)


class _Design:
    def __init__(self, dt, t, inputs):
        self.dt, self.t = np.asarray(dt), np.asarray(t)
        self.past, self.hold, self.x, self.preload = inputs
        self.base = np.empty((len(self.x), len(t) + 1, 7))
        self.base[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:] = self.x

    def make(self, taus, derivative=False):
        a = self.base.copy()
        if derivative:
            da = np.empty(a.shape[:2] + (2,))
        for m, tau in enumerate(taus):
            z = self.preload[:, m].copy()
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
    return _order_branches(p)


def _order_branches(p):
    if p[0] > p[1]:
        p[[0, 1, 2, 3]] = p[[1, 0, 3, 2]]
    if p[0] == p[1]:
        if p[1] < 8.:
            p[1] = np.nextafter(p[1], np.inf)
        else:
            p[0] = np.nextafter(p[0], -np.inf)
    return p[:2], p[2:]


def _fit_calibrated(train, checks, x, dt, t, cal, curves):
    rates = train['instant_rate_obs']
    values = _inputs(train, x, cal)
    labs = np.asarray(train['lab_id'])
    lab = np.asarray(checks['lab_id'])
    cmd, true = (np.asarray(checks[k]) for k in ('cmd_stress', 'true_stress'))
    sparse = []
    for c in np.unique(labs):
        ok = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        a, b = cmd[ok], true[ok]
        support = abs(b - (cal[c, 0] * a + cal[c, 1])) <= 7e-4
        if support.sum() < 2 or np.ptp(a[support]) < .1:
            sparse.append((c, a, b))
    if not sparse:
        return _fit(_Design(dt, t, values), curves, rates)
    pilot = ~np.isin(labs, [c for c, _, _ in sparse])
    if not pilot.any():
        return _fit(_Design(dt, t, values), curves, rates)
    base = _Design(dt, t, values)
    tau, theta = _fit(_Design(dt, t, tuple(a[pilot] for a in values)), curves[pilot], rates[pilot])
    y = np.column_stack((curves, rates))
    r = (base.make(tau) @ theta - y)[pilot]
    scale = np.maximum(.001, 2.5 * 1.4826 * np.median(abs(r - np.median(r, axis=0)), axis=0))
    blocks = []
    for c, a, b in sparse:
        m = labs == c
        past = train['past_stress_cmd'][m]
        hold = train['stress_hold_cmd'][m]
        z = np.zeros_like(values[3][m])
        gain = _Design(dt, t, (past, hold, values[2][m], z))
        bias = _Design(dt, t, (np.ones_like(past), np.ones_like(hold), values[2][m], z))
        blocks.append((m, gain, bias, a, b - (cal[c, 0] * a + cal[c, 1])))

    def residual(p):
        model = base.make(p[:2]) @ p[2:9]
        extras = []
        for (m, gain, bias, a, check), (dg, db) in zip(blocks, p[9:].reshape(-1, 2)):
            correction = dg * gain.make(p[:2])[:, :, :2] + db * bias.make(p[:2])[:, :, :2]
            model[m] += correction @ p[2:4]
            extras.extend(((dg * a + db - check) / .0007, np.array([db / .005])))
        return np.concatenate((((model - y) / scale).ravel(), *extras))

    n = len(sparse)
    result = least_squares(residual, np.r_[tau, theta, np.zeros(2*n)], loss='cauchy', x_scale='jac',
                           bounds=(np.r_[.2, .2, np.full(7, -np.inf), np.tile([-.2, -.15], n)],
                                   np.r_[8., 8., np.full(7, np.inf), np.tile([.2, .15], n)]),
                           max_nfev=100, ftol=1e-8)
    for (c, _, _), change in zip(sparse, result.x[9:].reshape(-1, 2)):
        cal[c] += change
    return _order_branches(result.x[:9])


def _allocate(dataset, delta):
    Z = np.asarray(dataset['Z_nuisance'], float)
    s0 = np.asarray(dataset['S_alloc_modes'], float) * (1.0 + 0.25 * np.tanh(delta))[:, None]
    y0 = np.asarray(dataset['y_alloc_obs'], float).ravel()
    P = np.linalg.solve(Z.T @ Z + float(dataset['nuisance_ridge']) * np.eye(Z.shape[1]), Z.T)
    S, y = s0 - Z @ (P @ s0), y0 - Z @ (P @ y0)
    ridge, u, b = (float(dataset[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    rhs = S.T @ y

    def gradient(w):
        return H @ w - rhs

    def value(w):
        r = S @ w - y
        return 0.5 * (r @ r + ridge * (w @ w))

    if b == 0 or b == m * u:
        w = np.full(m, b / m)
    else:
        scale = np.max(np.diag(H))
        result = minimize(lambda w: value(w) / scale, np.full(m, b/m),
                          jac=lambda w: gradient(w) / scale,
                          bounds=[(0., u)] * m,
                          constraints=[dict(type='eq', fun=lambda w: w.sum() - b,
                                            jac=lambda w: np.ones(m))],
                          method='SLSQP', options=dict(ftol=1e-14, maxiter=250))
        w = np.clip(result.x, 0., u)
        low, high = w <= 1e-8, w >= u - 1e-8
        free = ~(low | high)
        # Recover the exact equality-constrained optimum on the identified face.
        if free.any():
            k = free.sum()
            ones = np.ones((k, 1))
            K = np.block([[H[np.ix_(free, free)], -ones], [ones.T, np.zeros((1, 1))]])
            sol = np.linalg.solve(K, np.r_[rhs[free] - u * H[np.ix_(free, high)].sum(axis=1), b - u * high.sum()])
            if np.all((sol[:-1] >= -1e-11) & (sol[:-1] <= u + 1e-11)):
                w[low], w[high], w[free] = 0., u, np.clip(sol[:-1], 0., u)
    g = S.T @ (S @ w - y) + ridge * w
    # At a vertex any multiplier between the largest filled and smallest empty gradient works.
    free = (w > 1e-9) & (w < u - 1e-9)
    if free.any():
        nu = float(np.mean(g[free]))
    else:
        lo = np.max(g[w > u/2], initial=-np.inf)
        hi = np.min(g[w <= u/2], initial=np.inf)
        nu = float((lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (lo if np.isfinite(lo) else hi))
    primal = float(value(w))
    dual = float(primal - g @ w + nu * b + u * np.minimum(0., g - nu).sum())
    return w, dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(g - nu, 0.),
                   lam_upper=np.maximum(nu - g, 0.), primal_objective=primal,
                   dual_lower_bound=dual)


def solve_alpha_m96_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    aligned = _align(train, test)
    n = len(train['lab_id'])
    calibration = _stress_calibration(dataset['calibration_load_cells'])
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, theta = _fit_calibrated(train, dataset['calibration_load_cells'], aligned[:n], dt, t, calibration, curves)
    pred = np.einsum('ntj,j->nt', _Design(dt, t, _inputs(test, aligned[n:], calibration)).make(taus), theta)
    # For missing covariates the observed onset identifies the otherwise unobserved linear drift.
    onset = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1) & np.isfinite(onset)
    pred[missing, :-1] += (onset[missing] - pred[missing, -1])[:, None] * t
    w, cert = _allocate(dataset, pred[:, -2])
    return dict(estimated_taus=taus, predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                allocation_weights=w, kkt_certificate=cert)
