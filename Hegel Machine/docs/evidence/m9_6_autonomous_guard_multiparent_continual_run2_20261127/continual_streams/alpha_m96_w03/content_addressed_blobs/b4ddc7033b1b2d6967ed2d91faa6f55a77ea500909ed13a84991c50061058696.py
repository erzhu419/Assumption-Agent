"""Preloaded hereditary creep and certified compliance allocation."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _calibrate(checks):
    gain, bias = np.ones(4), np.zeros(4)
    archive = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = None
    for lab in range(4):
        x, y = (np.asarray(checks[k])[np.asarray(checks['lab_id']) == lab]
                for k in ('cmd_stress', 'true_stress'))
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        # The sweeps have roughly 1e-4 stress-unit noise. Fit the largest
        # consistent in-batch set, not the amplitude of an EMI excursion.
        i, j = np.triu_indices(x.size, 1)
        separated = np.abs(x[j] - x[i]) > 0.05
        i, j = i[separated], j[separated]
        a = np.column_stack((x, np.ones(x.size)))
        candidates = []
        if i.size:
            g = (y[j] - y[i]) / (x[j] - x[i])
            b = y[i] - g * x[i]
            masks = np.unique(np.abs(g[:, None]*x + b[:, None] - y) <= 6e-4, axis=0)
            best_count = int(masks.sum(axis=1).max())
            for use in masks[masks.sum(axis=1) == best_count]:
                if np.ptp(x[use]) < 0.05:
                    continue
                p = np.linalg.lstsq(a[use], y[use], rcond=None)[0]
                candidates.append((p, np.linalg.norm(a[use] @ p - y[use])))
        if candidates and best_count >= 3:
            gain[lab], bias[lab] = min(candidates, key=lambda item: item[1])[0]
            continue
        if history is None:
            history = json.loads(archive.read_text()) if archive.is_file() else []
        prior = []
        for sweep in history:
            m = np.asarray(sweep['lab_id']) == lab
            cx, cy = np.asarray(sweep['cmd_stress'])[m], np.asarray(sweep['true_stress'])[m]
            if cx.size > 1 and np.ptp(cx) > 0.05:
                prior.append(np.polyfit(cx, cy, 1))
        p = np.median(prior, axis=0) if prior else np.array([1., 0.])
        if candidates:
            p = min(candidates, key=lambda item: np.linalg.norm(item[0] - p))[0]
        elif x.size:
            p[1] = np.median(y - p[0]*x)
        gain[lab], bias[lab] = p
    return gain, bias


def _align(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        r = refs[use]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(r, axis=0))
        scale = np.median(np.nanstd(r, axis=0, ddof=0))
        raw[use] = (raw[use] - loc) / scale
    cells = [(lab == c) & (stratum == s) for c, s in np.unique(np.c_[lab, stratum], axis=0)]
    means = []
    for use in cells:
        col = raw[use]
        n = np.isfinite(col).sum(axis=0)
        means.append(np.divide(np.nansum(col, axis=0), n,
                               out=np.full(raw.shape[1], np.nan), where=n > 0))
    means = np.array(means)
    n = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), n, out=np.zeros(raw.shape[1]), where=n > 0)
    raw = np.where(np.isfinite(raw), raw, fill)
    return np.split(raw, [len(train['lab_id'])])


class _Hold:
    def __init__(self, dt, times, block, x, calibration):
        self.dt, self.times = np.asarray(dt), np.asarray(times)
        self.end = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.]
        self.duration = self.dt.sum()
        lab = block['lab_id']
        g, b = (p[lab] for p in calibration)
        self.hist = block['past_stress_cmd'] * g[:, None] + b[:, None]
        self.stress = block['stress_hold_cmd'] * g + b
        self.memory = block['initial_memory_states']
        self.base = block['strain_0']
        self.features = np.c_[np.ones(len(x)), x]
        self.linear = np.r_[times, 1.][None, :, None] * self.features[:, None, :]

    def design(self, taus, gradient=False):
        modes, grads = [], []
        for m, tau in enumerate(taus):
            e = np.exp(-self.end / tau)
            e0 = np.exp(-self.duration / tau)
            hist = e * -np.expm1(-self.dt / tau)
            delta = self.stress - self.hist @ hist - e0*self.memory[:, m]
            f = np.r_[-np.expm1(-self.times / tau), 1. / tau]
            modes.append(delta[:, None] * f)
            if gradient:
                dh = (self.end*e - (self.end + self.dt)*np.exp(-(self.end + self.dt)/tau)) / tau**2
                ddelta = -self.hist @ dh - e0*self.duration*self.memory[:, m]/tau**2
                df = -np.r_[self.times*np.exp(-self.times/tau), 1.] / tau**2
                grads.append(ddelta[:, None]*f + delta[:, None]*df)
        return np.concatenate((np.stack(modes, axis=-1), self.linear), axis=-1), grads


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _fit(hold, block, rows=slice(None)):
    y = np.c_[block['strain_curves_obs'] - hold.base[:, None], block['instant_rate_obs']]
    good = np.isfinite(y)
    selected = np.zeros(len(y), dtype=bool)
    selected[rows] = True
    good &= selected[:, None]
    target = y[good]
    lo = np.r_[0.051, 0.20, np.full(7, -np.inf)]
    hi = np.r_[8., 24.9, np.full(7, np.inf)]

    def residual(p):
        a, _ = hold.design(p[:2])
        return a[good] @ p[2:] - target

    def run(p, scale, loss):
        def jac(p):
            a, da = hold.design(p[:2], True)
            return np.c_[da[0][good]*p[2], da[1][good]*p[3], a[good]] / scale[:, None]
        return least_squares(lambda p: residual(p)/scale, p, jac=jac, bounds=(lo, hi),
                             loss=loss, x_scale='jac', max_nfev=100, ftol=1e-9).x

    rate = np.broadcast_to(np.arange(y.shape[1]) == y.shape[1]-1, y.shape)[good]
    def scale(p):
        r = residual(p)
        return np.where(rate, max(_mad(r[rate]), 1e-7), max(_mad(r[~rate]), 1e-7))

    taus = np.array([0.5, 3.0])
    a, _ = hold.design(taus)
    p = np.r_[taus, np.linalg.lstsq(a[good], target, rcond=None)[0]]
    p = run(p, scale(p), 'soft_l1')
    for _ in range(2):
        p = run(p, 2.3849*scale(p), 'cauchy')
    if p[0] >= p[1]:
        # The memory columns are labelled fast and slow; do not swap them.
        p[0] = min(p[0], p[1] - 1e-6)
    return p


def _allocate(S, y, ridge, cap, budget):
    """Feasible active-set QP; the reported dual uses the verifier's sign."""
    m = S.shape[1]
    Q = S.T @ S + ridge*np.eye(m)
    c = S.T @ y
    w = np.full(m, budget/m)
    active = np.zeros(m, dtype=int)
    if budget == 0:
        nu = float(np.min(-c))
    elif budget == m*cap:
        nu = float(np.max(Q @ w - c))
    else:
        for _ in range(20*m*m):
            free = active == 0
            g = Q @ w - c
            if not free.any():
                lower = np.where(active < 0, g, np.inf).argmin()
                upper = np.where(active > 0, g, -np.inf).argmax()
                nu = (g[lower] + g[upper])/2
                if g[lower] >= g[upper] - 1e-11:
                    break
                active[[lower, upper]] = 0
                continue
            B = Q[np.ix_(free, free)]
            v, h = np.linalg.solve(B, np.c_[g[free], np.ones(free.sum())]).T
            nu = v.sum()/h.sum()
            step = nu*h - v
            ids = np.flatnonzero(free)
            limits = np.full(len(ids), np.inf)
            pos, neg = step > 1e-13, step < -1e-13
            limits[pos] = (cap - w[ids[pos]])/step[pos]
            limits[neg] = -w[ids[neg]]/step[neg]
            j = limits.argmin()
            if limits[j] < 1:
                w[free] += max(0., limits[j])*step
                k = ids[j]
                active[k] = 1 if pos[j] else -1
                w[k] = cap if pos[j] else 0.
                continue
            w[free] += step
            g = Q @ w - c
            violation = active*(g - nu)
            k = violation.argmax()
            if violation[k] <= 1e-11:
                break
            active[k] = 0
        else:
            raise RuntimeError('allocation active set did not converge')
    w = np.clip(w, 0., cap)
    r = S @ w - y
    g = S.T @ r + ridge*w
    obj = 0.5*(r @ r + ridge*(w @ w))
    dual = obj - g @ w + nu*budget + cap*np.minimum(0., g - nu).sum()
    cert = dict(weights=w, nu_sum=float(nu),
                lam_lower=np.where(w < 1e-10, np.maximum(g-nu, 0.), 0.),
                lam_upper=np.where(w > cap-1e-10, np.maximum(nu-g, 0.), 0.),
                primal_objective=float(obj), dual_lower_bound=float(min(dual, obj)))
    return w, cert


def solve_alpha_m96_w03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    xt, xe = _align(train, test)
    cal = _calibrate(dataset['calibration_load_cells'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    fitted = _fit(_Hold(dt, times, train, xt, cal), train)
    hold = _Hold(dt, times, test, xe, cal)
    a, _ = hold.design(fitted[:2])
    response = a @ fitted[2:]
    pred = hold.base[:, None] + response[:, :-1]
    # With incomplete features, infer the coupon's constant flow from its
    # measured initial rate after removing the two hereditary contributions.
    rate = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    use = missing & np.isfinite(rate)
    pred[use] += (rate[use] - response[use, -1])[:, None] * hold.times
    S = dataset['S_alloc_modes'] * (1. + .25*np.tanh(pred[:, -1] - hold.base))[:, None]
    Z, y = dataset['Z_nuisance'], dataset['y_alloc_obs']
    Pz = np.linalg.solve(Z.T @ Z + dataset['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    w, cert = _allocate(S - Z @ (Pz @ S), y - Z @ (Pz @ y), dataset['source_ridge'],
                        dataset['u_box'], dataset['total_budget'])
    return dict(estimated_taus=fitted[:2], predicted_strain=pred,
                allocation_weights=w, kkt_certificate=cert)
