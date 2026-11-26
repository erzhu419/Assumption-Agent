"""Reference-panel calibration and certified symmetric-box sparse regression."""

import heapq
from itertools import count

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize


def _panel(panel):
    x = np.array(panel['X_panel_raw'], dtype=np.float64, copy=True)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    strata = np.asarray(panel['stratum'])
    _, centers = np.unique(panel['center_id'], return_inverse=True)
    cells = 3 * centers + strata.astype(int)
    w = len(x) / (len(np.bincount(cells)) * np.bincount(cells)[cells])
    for k in range(centers.max() + 1):
        m = centers == k
        c = controls[m]
        c = c[:, (np.isfinite(c).sum(axis=0) >= 2)]
        c = np.where(np.isfinite(c), c, np.nan)
        x[m] = (x[m] - np.median(np.nanmean(c, axis=0))) / np.median(np.nanstd(c, axis=0))
    valid = np.isfinite(x)
    x = np.where(valid, x, 0.0)
    means = (w @ x) / (w @ valid.astype(float))
    return np.where(valid, x, means), strata


def _cauchy(h, y):
    beta = np.linalg.lstsq(h, y, rcond=None)[0]
    for k in range(100):
        r = y - h @ beta
        # Let the broad initial fit narrow to the uncontaminated residual scale.
        mad = np.median(np.abs(r - np.median(r)))
        scale = max(1e-8, (2.0 if k < 4 else 1.5) * mad)
        a = 1.0 / np.hypot(1.0, r / scale)
        next_beta = np.linalg.lstsq(h * a[:, None], y * a, rcond=None)[0]
        if k >= 4 and np.linalg.norm(next_beta - beta) < 1e-10 * (1 + np.linalg.norm(beta)):
            return next_beta
        beta = next_beta
    return beta


def _diagonal(g):
    # Scale out the column norms, then maximize the geometric mean of the
    # diagonal split on the positive-definite interior with a log barrier.
    norms = np.sqrt(g.diagonal())
    cor = g / norms[:, None] / norms
    p = len(g)
    ix = np.diag_indices(p)
    ev = np.linalg.eigvalsh(cor)[0]
    if ev <= 1e-10:
        return np.zeros(p)
    d = np.full(p, ev * 0.5)
    a = cor.copy()
    for mu in (1.0, 0.25, 0.06, 0.015, 0.003):
        for _ in range(18):
            a[ix] = 1 - d
            chol = np.linalg.cholesky(a)
            inv = solve(a, np.eye(p), assume_a='pos', check_finite=False)
            grad = 1 / d - mu * inv.diagonal()
            step = solve(np.diag(1 / d**2) + mu * inv**2, grad, assume_a='pos', check_finite=False)
            gain = grad @ step
            if gain < 1e-9:
                break
            base = np.log(d).sum() + 2 * mu * np.log(chol.diagonal()).sum()
            t = 1.0
            while t > 1e-9:
                trial = d + t * step
                if np.all(trial > 0):
                    a[ix] = 1 - trial
                    try:
                        chol = np.linalg.cholesky(a)
                    except np.linalg.LinAlgError:
                        pass
                    else:
                        val = np.log(trial).sum() + 2 * mu * np.log(chol.diagonal()).sum()
                        if val >= base + 0.01 * t * gain:
                            d = trial
                            break
                t *= 0.5
    return (1 - 1e-9) * d * norms**2


def _box_l0(x, y, lam2, lam0, M):
    g, b = x.T @ x, x.T @ y
    const = 0.5 * (y @ y)
    p = len(b)
    ix = np.diag_indices(p)
    ridge = g.copy()
    ridge[ix] += lam2
    d = _diagonal(g)
    A = g.copy()
    A[ix] -= d
    q = d + lam2
    if lam0 > 0:
        t = np.sqrt(np.divide(2 * lam0, q, out=np.full(p, np.inf), where=q > 0))
        t = np.minimum(t, M)
        slope = lam0 / t + 0.5 * q * t
    best = np.zeros(p)
    ub = const
    tol = 2e-7
    leaves = []
    queue = []
    serial = count()

    def incumbent(mask):
        nonlocal best, ub
        i = np.flatnonzero(mask)
        if not len(i):
            return
        Q = ridge[np.ix_(i, i)]
        h = b[i]
        try:
            v = solve(Q, h, assume_a='pos', check_finite=False)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(Q, h, rcond=None)[0]
        if np.any(np.abs(v) > M):
            def fun(z):
                r = Q @ z
                return (0.5 * r - h) @ z, r - h
            v = minimize(fun, np.clip(v, -M, M), jac=True, method='L-BFGS-B',
                         bounds=[(-M, M)] * len(i),
                         options={'ftol': 1e-15, 'gtol': 1e-10}).x
        if lam0 > 0:
            v[np.abs(v) <= 1e-8] = 0
        z = np.zeros(p)
        z[i] = v
        residual = x @ z - y
        val = 0.5 * (residual @ residual + lam2 * (z @ z)) + lam0 * np.count_nonzero(z)
        if val < ub:
            ub, best = val, z

    def relax(zero, on, start):
        i = np.flatnonzero(~zero)
        active = on[i]
        q_i, t_i, s_i = q[i], t[i], slope[i]
        aa, hh = A[np.ix_(i, i)], b[i]
        # Split signs so that the free-variable convex envelope is continuously
        # differentiable on the nonnegative orthant.
        k = len(i)
        def fun(z):
            v = z[:k] - z[k:]
            r = aa @ v
            h = r - hh
            a, c = z[:k], z[k:]
            def side(u):
                g1 = np.where(active, q_i * u, np.where(u <= t_i, s_i, q_i * u))
                v1 = np.where(active, 0.5 * q_i * u**2,
                              np.where(u <= t_i, s_i * u, lam0 + 0.5 * q_i * u**2))
                return v1.sum(), g1
            va, da = side(a)
            vc, dc = side(c)
            return (0.5 * r - hh) @ v + va + vc, np.r_[h + da, -h + dc]

        # Already-active columns are a smooth signed quadratic. For these use
        # the norm of the difference, rather than a penalty on split parts.
        if active.any():
            base_fun = fun
            def fun(z):
                f, jac = base_fun(z)
                a, c = z[:k], z[k:]
                cross = q_i * active
                return f - (cross * a) @ c, jac - np.r_[cross * c, cross * a]

        w = np.zeros(p)
        if k:
            v = start[i]
            z = np.r_[np.maximum(v, 0), np.maximum(-v, 0)]
            fit = minimize(fun, z, jac=True, bounds=[(0.0, M)] * (2*k), method='L-BFGS-B',
                           options={'ftol': 2e-15, 'gtol': 1e-10, 'maxiter': 400})
            w[i] = fit.x[:k] - fit.x[k:]
        return w

    def bound(zero, on, w):
        r = A @ w
        h = r - b
        abs_h = np.abs(h)
        v = np.minimum(M, np.divide(abs_h, q, out=np.full(p, M), where=q > 0))
        m = lam0 + (0.5 * q * v - abs_h) * v
        free = ~(zero | on)
        lb = const - 0.5 * (w @ r) + m[on].sum() + np.minimum(m[free], 0.0).sum()
        return float(lb), m

    def leaf(zero, on, w, lb):
        leaves.append({'fixed_zero': zero.tolist(), 'fixed_active': on.tolist(),
                       'minorant_diagonal': d.tolist(), 'minorant_center': w.tolist(), 'lb': lb})

    def visit(zero, on, start):
        lb, m = bound(zero, on, start)
        if lb >= ub - tol:
            leaf(zero, on, start, lb)
            return
        free = ~(zero | on)
        w = relax(zero, on, start)
        v = np.abs(w)
        if not free.any() or np.all((v[free] < 1e-7) | (v[free] >= t[free] - 1e-7)):
            incumbent(on | (free & (v > 1e-7)))
        lb, m = bound(zero, on, w)
        if lb >= ub - tol:
            leaf(zero, on, w, lb)
            return
        if not free.any():
            # Exact box-QP solution supplies the tangent point if the numerical
            # envelope solution stopped early on a fully assigned support.
            i = np.flatnonzero(on)
            if len(i):
                Q, h = ridge[np.ix_(i, i)], b[i]
                def fun(v):
                    r = Q @ v
                    return (0.5 * r - h) @ v, r - h
                w[i] = minimize(fun, w[i], jac=True, method='L-BFGS-B',
                                bounds=[(-M, M)] * len(i),
                                options={'ftol': 1e-16, 'gtol': 1e-12, 'maxiter': 500}).x
            lb, _ = bound(zero, on, w)
            leaf(zero, on, w, lb)
            return
        fraction = np.minimum(v / t, 1.0)
        j = np.argmax(np.where(free, fraction * (1 - fraction) + 1e-5 / (1 + np.abs(m)), -1))
        heapq.heappush(queue, (lb, next(serial), zero, on, w, j))

    no = np.zeros(p, dtype=bool)
    if lam0 == 0:
        # With no support penalty the root is the box-ridge problem itself.
        incumbent(~no)
        lb, _ = bound(no, no, best)
        leaf(no, no, best, lb)
    else:
        start = relax(no, no, np.zeros(p))
        # Threshold the root relaxation to seed sparse feasible supports.
        ratio = np.abs(start) / t
        for cutoff in (1e-6, 0.10, 0.25, 0.50, 0.75, 1.0 - 1e-6):
            incumbent(ratio >= cutoff)
        visit(no, no, start)
    while queue:
        lb, _, zero, on, w, j = heapq.heappop(queue)
        if lb >= ub - tol:
            leaf(zero, on, w, lb)
        else:
            z, a = zero.copy(), on.copy()
            z[j] = a[j] = True
            visit(z, on, w)
            visit(zero, a, w)
    return {'weights': best.tolist(), 'objective': float(ub),
            'certified_lower_bound': min(float(ub), min(leaf['lb'] for leaf in leaves)),
            'subcube_leaves': leaves}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    aligned, strata = _panel(study['panel'])
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    reference = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    demod = np.angle(signal * reference.conj())
    h = np.column_stack((np.ones(len(aligned)), strata == 1, strata == 2, aligned, demod))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal, s3 = np.asarray(study['calibration_indices']), np.asarray(study['stage3_indices'])
    beta = _cauchy(h[cal], y[cal])
    target = y[s3] - h[s3] @ beta
    cert = _box_l0(np.asarray(study['X_cand'], dtype=np.float64), target,
                   float(study['lam2']), float(study['lam0']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': beta,
            'deconfounded_target': target, 'certified_selection': cert}
