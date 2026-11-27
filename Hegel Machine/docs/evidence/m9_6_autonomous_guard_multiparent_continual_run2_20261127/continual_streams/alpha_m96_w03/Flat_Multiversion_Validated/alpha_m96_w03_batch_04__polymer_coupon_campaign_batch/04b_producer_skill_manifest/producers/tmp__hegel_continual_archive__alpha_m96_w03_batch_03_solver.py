"""Preloaded two-branch hereditary creep and certified compliance allocation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _calibrate(checks):
    gain = np.ones(4)
    bias = np.zeros(4)
    for i in range(4):
        m = np.asarray(checks['lab_id']) == i
        x = np.asarray(checks['cmd_stress'])[m]
        y = np.asarray(checks['true_stress'])[m]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        a, b = np.triu_indices(len(x), 1)
        sep = np.abs(x[a] - x[b]) > 1e-10
        a, b = a[sep], b[sep]
        if not len(a):
            if len(x):
                bias[i] = np.median(y - x)
            continue
        g = np.median((y[a] - y[b]) / (x[a] - x[b]))
        r = y - g*x
        keep = np.abs(r - np.median(r)) <= max(4 * _mad(r), 1e-8)
        if np.ptp(x[keep]) > 1e-10:
            gain[i], bias[i] = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
        else:
            gain[i], bias[i] = g, np.median(r)
    return gain, bias


def _align(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    out = np.empty_like(raw, dtype=float)
    for i in np.unique(lab):
        m = lab == i
        ref_i = ref[m]
        ref_i = ref_i[:, np.isfinite(ref_i).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(ref_i, axis=0))
        scale = np.median(np.nanstd(ref_i, axis=0, ddof=0))
        out[m] = (raw[m] - loc) / scale
    # Each observed (lab, stratum) cell has one vote in the pooled fallback.
    cells = [(lab == i) & (stratum == s) for i, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    finite = np.isfinite(out)
    means = []
    for m in cells:
        count = finite[m].sum(axis=0)
        means.append(np.divide(np.nansum(out[m], axis=0), count, out=np.full(out.shape[1], np.nan), where=count > 0))
    means = np.array(means)
    fallback = np.nanmean(means, axis=0)
    out = np.where(finite, out, fallback)
    n = len(train['lab_id'])
    return out[:n], out[n:]


class _Hold:
    def __init__(self, interval, times, block, x, calibration):
        self.times = times
        self.dt = interval
        self.end = np.r_[np.cumsum(interval[:0:-1])[::-1], 0.]
        lab = block['lab_id']
        g, b = (c[lab] for c in calibration)
        self.hist = np.asarray(block['past_stress_cmd']) * g[:, None] + b[:, None]
        self.stress = np.asarray(block['stress_hold_cmd']) * g + b
        self.features = np.column_stack((np.ones(len(x)), x))
        self.base = np.asarray(block['strain_0'])
        self.memory = np.asarray(block['initial_memory_states'])
        self.history_duration = self.dt.sum()
        self.t = np.r_[times, 1.]
        self.linear = self.features[:, None, :] * self.t[None, :, None]

    def design(self, taus, gradient=False):
        matrices, derivs = [], []
        for j, tau in enumerate(taus):
            e1 = np.exp(-self.end / tau)
            e2 = np.exp(-(self.end + self.dt) / tau)
            w = e1 * -np.expm1(-self.dt / tau)
            preload = self.memory[:, j] * np.exp(-self.history_duration / tau)
            delta = self.stress - np.einsum('ij,j->i', self.hist, w) - preload
            mode = np.r_[-np.expm1(-self.times / tau), 1. / tau]
            matrices.append(delta[:, None] * mode)
            if gradient:
                dw = (self.end * e1 - (self.end + self.dt) * e2) / tau**2
                dd = -np.einsum('ij,j->i', self.hist, dw) - preload * self.history_duration / tau**2
                dm = -np.r_[self.times*np.exp(-self.times/tau), 1.] / tau**2
                derivs.append(dd[:, None] * mode + delta[:, None] * dm)
        design = np.concatenate((np.stack(matrices, axis=-1), self.linear), axis=-1)
        return design, derivs


def _fit(hold, block):
    curves = np.asarray(block['strain_curves_obs']) - hold.base[:, None]
    rate = np.asarray(block['instant_rate_obs'])
    obs = np.column_stack((curves, rate))
    good = np.isfinite(obs)
    y = obs[good]
    lo = np.r_[[.2, .2], np.full(7, -np.inf)]
    hi = np.r_[[8., 8.], np.full(7, np.inf)]

    def optimize(p, sigma, loss):
        w = np.broadcast_to(1. / sigma, obs.shape)[good]
        def residual(v):
            a, _ = hold.design(v[:2])
            return (np.einsum('ijk,k->ij', a, v[2:])[good] - y) * w
        def jacobian(v):
            a, ds = hold.design(v[:2], gradient=True)
            return np.column_stack((ds[0][good]*v[2], ds[1][good]*v[3], a[good])) * w[:, None]
        return least_squares(residual, p, jac=jacobian, bounds=(lo, hi), loss=loss, x_scale='jac', max_nfev=130, ftol=1e-9)

    p0 = np.array([.5, 3.])
    a, _ = hold.design(p0)
    theta = np.linalg.lstsq(a[good], y, rcond=None)[0]
    err = obs - np.einsum('ijk,k->ij', a, theta)
    k = curves.shape[1]
    pilot_scales = np.r_[np.full(k, max(_mad(err[:, :-1][good[:, :-1]]), 1e-7)), max(_mad(err[:, -1][good[:, -1]]), 1e-7)]
    p = optimize(np.r_[p0, theta], pilot_scales, 'soft_l1').x
    for _ in range(2):
        a, _ = hold.design(p[:2])
        err = obs - np.einsum('ijk,k->ij', a, p[2:])
        scales = np.r_[np.full(k, max(_mad(err[:, :-1][good[:, :-1]]), 1e-7)), max(_mad(err[:, -1][good[:, -1]]), 1e-7)]
        p = optimize(p, 2.3849 * scales, 'cauchy').x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def _resolve_calibration(dataset, features):
    checks = dataset['calibration_load_cells']
    cal = np.array(_calibrate(checks))
    lab, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    train = dataset['train']
    labels = np.asarray(train['lab_id'])
    uncertain = {}
    for j in range(4):
        use = (lab == j) & np.isfinite(x) & np.isfinite(y)
        a, b = x[use], y[use]
        cmd = np.concatenate([block[key][block['lab_id'] == j].ravel()
                              for block in (train, dataset['test'])
                              for key in ('past_stress_cmd', 'stress_hold_cmd')])
        if not cmd.size:
            continue
        span = float(np.ptp(cmd))
        tol = 0.001  # clean archival sweep deviations are below 3e-4
        base = cal[:, j].copy()
        residual = np.abs(b - base[0]*a - base[1])
        good = residual <= tol
        if good.sum() >= 4 and np.ptp(a[good]) >= 0.25*span and good.sum() == len(a):
            continue
        pairs = [base]
        for i, k in zip(*np.triu_indices(len(a), 1)):
            if abs(a[k] - a[i]) > max(1e-10, 0.15*span):
                gain = (b[k] - b[i])/(a[k] - a[i])
                if gain > 0:
                    pairs.append(np.array([gain, b[i] - gain*a[i]]))
        design = np.c_[a, np.ones_like(a)]
        groups = {}
        for gb in pairs:
            mask = np.abs(b - design @ gb) <= tol
            if mask.sum() >= 2 and np.ptp(a[mask]) > max(1e-10, 0.15*span):
                gb = np.linalg.lstsq(design[mask], b[mask], rcond=None)[0]
                if gb[0] > 0:
                    groups[tuple(mask)] = gb
        if groups:
            count = max(map(sum, groups))
            choices = [gb for mask, gb in groups.items() if sum(mask) == count]
            limits = np.array([cmd.min(), cmd.max()])
            effect = (np.asarray(choices) - choices[0]) @ np.array([limits, np.ones(2)])
            if count >= 3 and count > len(a)/2 and np.max(np.abs(effect)) <= .01:
                cal[:, j] = choices[0]
                continue
        uncertain[j] = (a, b, list(groups.values()) or [base], np.array([cmd.min(), cmd.max()]))
    if not uncertain:
        return cal

    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = []
    if path.is_file():
        with path.open() as f:
            history = [np.array(_calibrate(sweep)) for sweep in json.load(f)]
    donors = ~np.isin(labels, list(uncertain))
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    def block(mask):
        return {key: np.asarray(a)[mask] for key, a in train.items()}
    p = None
    if donors.any():
        subset = block(donors)
        p = _fit(_Hold(dt, times, subset, features[donors], cal), subset)
    for j, (a, b, current, limits) in uncertain.items():
        priors = [s[:, j] for s in history]
        options = current + priors
        for g, _ in priors:
            options += [np.array([g, v]) for v in b - g*a]
        target = labels == j
        if p is None or not target.any():
            if len(a):
                cal[:, j] = min(options, key=lambda gb: (-np.sum(np.abs(b-gb[0]*a-gb[1]) < .001),
                                                          np.median(np.abs(b-gb[0]*a-gb[1]))))
            elif priors:
                cal[:, j] = priors[-1]
            continue
        subset = block(target)
        hold = _Hold(dt, times, subset, features[target], cal)
        obs = np.c_[subset['strain_curves_obs'] - hold.base[:, None], subset['instant_rate_obs']]
        # Compare affine maps on this lab using coefficients fitted only to trusted labs.
        def coupon_losses(gb):
            g, bias = gb
            hold.hist = subset['past_stress_cmd'] * g + bias
            hold.stress = subset['stress_hold_cmd'] * g + bias
            err = np.abs(hold.design(p[:2])[0] @ p[2:] - obs)
            curve = np.ma.median(np.ma.masked_invalid(err[:, :-1]), axis=1).filled(np.nan)
            rate = err[:, -1]
            return np.where(np.isfinite(curve), curve, 0.) + np.where(np.isfinite(rate), rate, 0.)

        maps = np.asarray(options)
        counts = np.sum(np.abs(b - maps[:, :1]*a - maps[:, 1:]) <= .001, axis=1)
        # Require spot-check support before comparing competing maps. Where the
        # curve and rate cannot distinguish them, favor continuity of the map.
        eligible = counts >= max(0, counts.max() - 1)
        losses = np.array([coupon_losses(gb) for gb in maps])
        winner = np.argmin(np.where(eligible, np.median(losses, axis=1), np.inf))
        if priors:
            delta = losses - losses[winner]
            center = np.median(delta, axis=1)
            spread = 1.4826*np.median(np.abs(delta - center[:, None]), axis=1)
            tied = eligible & (center <= 2.5*spread/np.sqrt(delta.shape[1]) + 1e-5)
            distance = np.min(np.max(np.abs((maps[:, None] - priors) @
                                          np.array([limits, np.ones(2)])), axis=2), axis=1)
            winner = np.argmin(np.where(tied, distance, np.inf))
        cal[:, j] = maps[winner]
    return cal


def _allocate(dataset, strain):
    d = dataset
    z = np.asarray(d['Z_nuisance'], dtype=float)
    pz = np.linalg.solve(z.T @ z + d['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    gain = 1.0 + 0.25 * np.tanh(strain[:, -1] - d['test']['strain_0'])
    s = np.asarray(d['S_alloc_modes'], dtype=float) * gain[:, None]
    y = np.asarray(d['y_alloc_obs'], dtype=float).ravel()
    s, y = s - z @ (pz @ s), y - z @ (pz @ y)
    ridge, upper, budget = (float(d[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    n = s.shape[1]
    h = s.T @ s + ridge * np.eye(n)
    c = s.T @ y

    def objective(w):
        return (0.5 * (w @ h @ w) - c @ w, h @ w - c)

    result = minimize(objective, np.full(n, budget/n), jac=True, method='SLSQP',
                      bounds=[(0., upper)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                    'jac': lambda w: np.ones(n)}],
                      options={'ftol': 1e-13, 'maxiter': 250})
    w = np.clip(result.x, 0., upper)
    lo, hi = w < 1e-7, w > upper - 1e-7
    free = ~(lo | hi)
    if free.any():
        # On the identified face, solve the exact first-order system.
        nf = free.sum()
        system = np.block([[h[np.ix_(free, free)], -np.ones((nf, 1))],
                           [np.ones((1, nf)), np.zeros((1, 1))]])
        sol = np.linalg.solve(system, np.r_[c[free] - upper*h[np.ix_(free, hi)].sum(axis=1),
                                             budget - upper*hi.sum()])
        if np.all((sol[:-1] >= 0) & (sol[:-1] <= upper)):
            w[lo], w[hi], w[free] = 0., upper, sol[:-1]
    # Restore equality after bound clipping, then use exact pair exchanges
    # if the optimizer's active face did not satisfy the KKT inequalities.
    remaining = budget - w.sum()
    if remaining > 0:
        w[np.argmax(upper - w)] += remaining
    elif remaining < 0:
        w[np.argmax(w)] += remaining
    for _ in range(3000):
        g = h @ w - c
        i = np.argmin(np.where(w < upper, g, np.inf))
        j = np.argmax(np.where(w > 0, g, -np.inf))
        if g[j] - g[i] <= 2e-10 or w[i] >= upper or w[j] <= 0:
            break
        step = min(upper - w[i], w[j], (g[j] - g[i])/(h[i,i]+h[j,j]-2*h[i,j]))
        w[i] += step
        w[j] -= step
    res = s @ w - y
    g = s.T @ res + ridge*w
    # The tangent bound is maximized at a gradient order statistic.
    order = max(0, min(n-1, int(np.ceil(budget/upper))-1))
    nu = float(np.sort(g)[order])
    primal = float(0.5*(res@res + ridge*(w@w)))
    dual = float(primal - g@w + nu*budget + upper*np.minimum(g-nu, 0.).sum())
    cert = {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': np.maximum(g-nu, 0.),
            'lam_upper': np.maximum(nu-g, 0.), 'primal_objective': primal,
            'dual_lower_bound': min(primal, dual)}
    return w, cert


def solve_alpha_m96_w03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _align(train, test)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    cal = _resolve_calibration(dataset, x_train)
    train_hold = _Hold(dt, time, train, x_train, cal)
    p = _fit(train_hold, train)
    test_hold = _Hold(dt, time, test, x_test, cal)
    pred = test_hold.base[:, None] + test_hold.design(p[:2])[0][:, :-1] @ p[2:]
    weights, cert = _allocate(dataset, pred)
    return {'estimated_taus': p[:2], 'predicted_strain': pred,
            'allocation_weights': weights, 'kkt_certificate': cert}
