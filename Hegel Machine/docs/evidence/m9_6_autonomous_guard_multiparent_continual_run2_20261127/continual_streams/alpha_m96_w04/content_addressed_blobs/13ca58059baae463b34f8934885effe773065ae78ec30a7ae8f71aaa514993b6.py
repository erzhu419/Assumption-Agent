import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _mad(x, axis=None):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _calibration(checks):
    def rows(record, c):
        lab, cmd, true = (np.asarray(record[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
        mask = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        return cmd[mask], true[mask]

    def fit(x, y):
        i, j = np.triu_indices(x.size, 1)
        dx = x[i] - x[j]
        keep = abs(dx) > max(1e-8, 0.1 * np.ptp(x))
        gain = np.median((y[i[keep]] - y[j[keep]]) / dx[keep])
        bias = np.median(y - gain * x)
        A = np.column_stack((x, np.ones(x.size)))
        scale = max(2 * _mad(y - A @ [gain, bias]), 2e-4)
        return least_squares(lambda p: A @ p - y, [gain, bias], jac=lambda p: A.copy(),
                             loss='cauchy', f_scale=scale).x

    def consensus(x, y):
        # Each well-separated pair proposes an affine line; require support
        # from a third check and a majority before overriding the archive.
        if x.size < 3 or np.ptp(x) < 0.1:
            return None
        i, j = np.triu_indices(x.size, 1)
        keep = abs(x[i] - x[j]) >= 0.1
        i, j = i[keep], j[keep]
        gain = (y[i] - y[j]) / (x[i] - x[j])
        bias = y[i] - gain * x[i]
        error = abs(y[None, :] - gain[:, None] * x[None, :] - bias[:, None])
        inside = error < 0.002
        count = inside.sum(axis=1)
        best = np.lexsort((np.minimum(error, 0.002).sum(axis=1), -count))[0]
        mask = inside[best]
        if count[best] >= max(3, x.size // 2 + 1) and np.ptp(x[mask]) >= 0.1:
            return fit(x[mask], y[mask])
        return None

    coef = np.empty((4, 2))
    history = None
    for c in range(4):
        x, y = rows(checks, c)
        p = consensus(x, y)
        if p is not None:
            coef[c] = p
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text()) if _HISTORY.is_file() else []
        priors = []
        for sweep in history:
            a, b = rows(sweep, c)
            p = consensus(a, b)
            if p is not None:
                priors.append(p)
        prior = np.median(priors, axis=0) if priors else np.array([1., 0.])
        # With only two agreeing checks, use their current slope only if
        # consistent with the archive; tightly spaced pairs cannot identify it.
        residual = y - (prior[0] * x + prior[1])
        i, j = np.triu_indices(x.size, 1)
        dx = abs(x[i] - x[j])
        candidates = (dx >= 0.1) & (abs(residual[i] - residual[j]) < 0.02 * dx + 0.002)
        if np.any(candidates):
            score = np.where(candidates, abs(residual[i] - residual[j]) / np.maximum(dx, 0.1)
                             + 0.5 * (abs(residual[i]) + abs(residual[j])), np.inf)
            k = np.argmin(score)
            if max(abs(residual[i[k]]), abs(residual[j[k]])) < 0.02:
                take = [i[k], j[k]]
                coef[c] = fit(x[take], y[take])
                continue
        near = residual[abs(residual) < 0.02]
        coef[c] = prior + [0., np.median(near) if near.size else 0.]
    return coef


def _aligned(train, test):
    lab, stratum, raw, ref = [np.concatenate((train[k], test[k])) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')]
    features = raw.copy()
    for c in np.unique(lab):
        m = lab == c
        loc = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        features[m] = (raw[m] - loc) / scale
    observed = np.isfinite(features)
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        m = (lab == c) & (stratum == s)
        count = observed[m].sum(axis=0)
        means.append(np.divide(np.nansum(features[m], axis=0), count,
                               out=np.full(features.shape[1], np.nan), where=count > 0))
    return np.where(observed, features, np.nanmean(means, axis=0))


class _Creep:
    def __init__(self, part, features, coef, dt, times):
        self.t = np.r_[times, 0.]
        self.suffix = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
        gain, bias = coef[np.asarray(part['lab_id'])].T
        self.past = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = part['stress_hold_cmd'] * gain + bias
        self.initial = np.asarray(part['initial_memory_states'])
        drift = np.column_stack((np.ones(len(features)), features))
        self.drift = drift[:, None, :] * np.r_[times, 1.][None, :, None]

    def branch(self, tau, mode):
        e = np.exp(-self.suffix / tau)
        de = e * self.suffix / tau**2
        # Exact piecewise-constant propagation from the recorded preload.
        q = self.hold - np.einsum('ij,j->i', self.past, np.diff(e)) - self.initial[:, mode] * e[0]
        dq = -np.einsum('ij,j->i', self.past, np.diff(de)) - self.initial[:, mode] * de[0]
        f = -np.expm1(-self.t / tau)
        df = -self.t * np.exp(-self.t / tau) / tau**2
        f[-1], df[-1] = 1. / tau, -1. / tau**2
        return q[:, None] * f, dq[:, None] * f + q[:, None] * df

    def basis(self, taus):
        a, da = self.branch(taus[0], 0)
        b, db = self.branch(taus[1], 1)
        return np.concatenate((a[..., None], b[..., None], self.drift), axis=-1), da, db


def _fit(model, targets):
    valid = np.isfinite(targets)
    y = targets[valid]
    # Optimize ordered times without ever reassigning a measured memory slot.
    def taus(p):
        return np.array([p[0], p[0] + (24.99 - p[0]) * p[1]])

    def run(start, scales):
        scale = np.broadcast_to(scales, targets.shape)[valid]

        def residual(p):
            a = model.basis(taus(p))[0][valid]
            return (np.einsum('ij,j->i', a, p[2:]) - y) / scale

        def jac(p):
            a, da, db = model.basis(taus(p))
            x, z = da[valid] * p[2], db[valid] * p[3]
            return np.column_stack((x + (1 - p[1]) * z, (24.99 - p[0]) * z, a[valid])) / scale[:, None]

        return least_squares(residual, start, jac=jac, loss='cauchy', x_scale='jac',
                             bounds=([.051, 1e-7] + [-np.inf]*7, [24.98, 1.] + [np.inf]*7),
                             max_nfev=130, ftol=1e-9)

    initial = []
    for t in ([.35, 2.], [.65, 4.], [1.2, 6.5], [.18, 7.], [1.8, 12.]):
        a = model.basis(t)[0]
        w = np.linalg.lstsq(a[valid], y, rcond=None)[0]
        initial.append(np.r_[t[0], (t[1] - t[0]) / (24.99 - t[0]), w])
    p = initial[1]
    r = model.basis(taus(p))[0] @ p[2:] - targets
    scales = np.ones(targets.shape[1])
    for j in np.flatnonzero(valid.any(axis=0)):
        scales[j] = max(2 * _mad(r[valid[:, j], j]), 1e-6)
    best = min((run(p, scales) for p in initial), key=lambda result: result.cost)
    for _ in range(2):
        p = best.x
        r = model.basis(taus(p))[0] @ p[2:] - targets
        curve = r[:, :-1][valid[:, :-1]]
        rate = r[:, -1][valid[:, -1]]
        scales = np.r_[np.repeat(max(2 * _mad(curve), 1e-6), r.shape[1] - 1),
                       max(2 * _mad(rate), 1e-6) if rate.size else 1.]
        best = run(p, scales)
    return taus(best.x), best.x[2:]


def _allocate(S, y, ridge, upper, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    b = S.T @ y
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1 lower, +1 upper
    for _ in range(8 * n * n):
        g = H @ w - b
        free = np.flatnonzero(active == 0)
        if free.size:
            # Equality-constrained Newton step on the current face.
            mat = H[np.ix_(free, free)]
            inv_g, inv_one = np.linalg.solve(mat, np.column_stack((g[free], np.ones(free.size)))).T
            nu = inv_g.sum() / inv_one.sum()
            step = nu * inv_one - inv_g
            moved = abs(step) > 1e-12
            if np.any(moved):
                limit = np.full(free.size, np.inf)
                down, up = step < -1e-12, step > 1e-12
                limit[down] = -w[free[down]] / step[down]
                limit[up] = (upper - w[free[up]]) / step[up]
                j = np.argmin(limit)
                w[free] += min(1., max(0., limit[j])) * step
                if limit[j] < 1.:
                    k = free[j]
                    active[k] = 1 if step[j] > 0 else -1
                    w[k] = upper if active[k] == 1 else 0.
                continue
        else:
            lo = np.max(g[active == 1], initial=-np.inf)
            hi = np.min(g[active == -1], initial=np.inf)
            nu = np.clip(0., min(lo, hi), max(lo, hi))
        violations = active * (g - nu)
        j = np.argmax(violations)
        if violations[j] <= 1e-10:
            break
        active[j] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')
    w = np.clip(w, 0., upper)
    residual = S @ w - y
    obj = 0.5 * (residual @ residual + ridge * (w @ w))
    g = S.T @ residual + ridge * w
    dual = obj - g @ w + nu * budget + upper * np.minimum(g - nu, 0.).sum()
    return w, dict(weights=w.copy(), nu_sum=float(nu), lam_lower=np.maximum(g - nu, 0.),
                   lam_upper=np.maximum(nu - g, 0.), primal_objective=float(obj),
                   dual_lower_bound=float(min(obj, dual)))


def solve_alpha_m96_w04_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    coef = _calibration(dataset['calibration_load_cells'])
    aligned = _aligned(train, test)
    n = len(train['lab_id'])
    model = _Creep(train, aligned[:n], coef, dt, times)
    targets = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                               train['instant_rate_obs']))
    taus, theta = _fit(model, targets)
    pred = _Creep(test, aligned[n:], coef, dt, times).basis(taus)[0][:, :-1] @ theta
    pred += test['strain_0'][:, None]
    S = dataset['S_alloc_modes'] * (1 + .25 * np.tanh(pred[:, -1] - test['strain_0']))[:, None]
    Z, y = np.asarray(dataset['Z_nuisance']), np.asarray(dataset['y_alloc_obs']).ravel()
    Pz = np.linalg.solve(Z.T @ Z + dataset['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    w, cert = _allocate(S - Z @ (Pz @ S), y - Z @ (Pz @ y), dataset['source_ridge'],
                        dataset['u_box'], dataset['total_budget'])
    return dict(estimated_taus=taus, predicted_strain=pred, allocation_weights=w, kkt_certificate=cert)
