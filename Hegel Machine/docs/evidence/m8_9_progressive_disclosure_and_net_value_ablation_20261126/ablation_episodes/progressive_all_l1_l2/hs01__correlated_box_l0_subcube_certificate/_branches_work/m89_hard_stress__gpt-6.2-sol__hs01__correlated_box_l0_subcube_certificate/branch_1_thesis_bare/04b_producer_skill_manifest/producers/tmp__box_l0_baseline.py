"""Certified box-constrained L0 regression by support enumeration.

The support table is small (at most 2**16 entries). Each entry has a
quadratic tangent certificate. Taking the minimum over the entries in a
subcube yields a certificate for that entire subcube, without a PSD split.
"""

import numpy as np
from scipy.optimize import lsq_linear


def solve_certified_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    tol = float(problem['gap_tol'])
    n, p = X.shape
    threshold = 1e-10
    tiny_box = min(box, threshold)
    if p == 0 or box <= threshold:
        w = np.zeros(p)
        if p:
            A = np.vstack((X, np.sqrt(ridge) * np.eye(p)))
            b = np.r_[y, np.zeros(p)]
            w = lsq_linear(A, b, bounds=(-box, box), method='bvls', tol=1e-13).x
        return _small_box_result(X, y, ridge, box, w)

    # Extended precision is for certificate evaluation, not for the solves.
    # G, c, and s represent the smooth objective .5*w'Gw - c'w + s.
    D = np.longdouble
    XL, yL = X.astype(D), y.astype(D)
    G = XL.T @ XL + D(ridge) * np.eye(p, dtype=D)
    c = XL.T @ yL
    s = D(0.5) * (yL @ yL)
    # Bound accumulation and evaluation roundoff using the absolute scale
    # of the unexpanded least-squares terms, including the gradient terms.
    row_scale = D(box) * np.sum(np.abs(XL), axis=1) + np.abs(yL)
    scale = np.sum(row_scale**2) + D(ridge)*p*D(box)**2 + D(p)*penalty
    margin = 32 * (n + p + 4) * np.finfo(D).eps * scale
    Gf, cf = G.astype(float), c.astype(float)
    # This stabilization only proposes tangent points. It is not used
    # when computing their lower bounds.
    reg = 1e-12 * max(np.max(np.diag(Gf)), 1e-100)
    stabilized = Gf + reg * np.eye(p)

    def evaluate(w, active):
        v = np.asarray(w, dtype=D)
        grad = v @ G - c
        q = s + D(0.5) * np.sum(v * (grad - c), axis=-1)
        # Convexity: f(u) >= f(v) + grad(v)'(u - v). Minimize the
        # plane over the support's box. Allow tiny inactive weights as
        # well, so this also bounds the thresholded objective in the API.
        radii = np.where(active, D(box), D(tiny_box))
        cardinality = np.count_nonzero(active, axis=-1)
        lower = (q - np.sum(v * grad, axis=-1)
                 - np.sum(radii * np.abs(grad), axis=-1)
                 + D(penalty) * cardinality - margin)
        upper = q + D(penalty) * np.count_nonzero(np.abs(w) > threshold, axis=-1)
        return lower, upper

    def objective(w):
        r = X @ w - y
        return float(0.5 * (r @ r + ridge * (w @ w))
                     + penalty * np.count_nonzero(np.abs(w) > threshold))

    N = 1 << p
    bits = 1 << np.arange(p)
    lower = np.empty(N, dtype=D)
    best = np.zeros(p)
    incumbent = objective(best)
    for start in range(0, N, 2048):
        stop = min(N, start + 2048)
        active = (np.arange(start, stop)[:, None] & bits) != 0
        mat = stabilized * (active[:, :, None] & active[:, None, :])
        idx = np.arange(p)
        mat[:, idx, idx] += ~active
        w = np.linalg.solve(mat, (active * cf)[..., None])[..., 0]
        np.clip(w, -box, box, out=w)
        lower[start:stop], upper = evaluate(w, active)
        candidate = int(np.argmin(upper))
        value = objective(w[candidate])
        if value < incumbent:
            incumbent, best = value, w[candidate].copy()

    A = np.vstack((X, np.sqrt(ridge) * np.eye(p))) if ridge else X
    b = np.r_[y, np.zeros(p)] if ridge else y
    cutoff = tol * 0.9
    for mask in np.argsort(lower):
        if lower[mask] >= incumbent - cutoff:
            break
        active = (mask & bits) != 0
        w = np.zeros(p)
        if np.any(active):
            fit = lsq_linear(A[:, active], b, bounds=(-box, box),
                             method='bvls', tol=1e-13)
            w[active] = np.clip(fit.x, -box, box)
        value = objective(w)
        if value < incumbent:
            incumbent, best = value, w.copy()
        bound, _ = evaluate(w, active)
        lower[mask] = max(lower[mask], bound)
        if lower[mask] < incumbent - cutoff and np.any(~active):
            # Usually the tiny inactive interval needs no optimization.
            # If its possible gain matters, solve the expanded box too.
            radii = np.where(active, box, tiny_box)
            fit = lsq_linear(A, b, bounds=(-radii, radii), method='bvls', tol=1e-13)
            w = np.clip(fit.x, -radii, radii)
            value = objective(w)
            if value < incumbent:
                incumbent, best = value, w.copy()
            bound, _ = evaluate(w, active)
            lower[mask] = max(lower[mask], bound)

    # A support-wise lower bound remains valid when inactive coordinates
    # are restricted to zero as in the leaf definition. The minimum over
    # all supports therefore certifies the whole lattice as one subcube.
    lb = max(0.0, float(np.nextafter(float(lower.min()), -np.inf)))
    return _result(best, incumbent, min(incumbent, lb))


def _small_box_result(X, y, ridge, box, w):
    D = np.longdouble
    A, b, v = X.astype(D), y.astype(D), w.astype(D)
    r = A @ v - b
    grad = A.T @ r + D(ridge)*v
    q = (r @ r + D(ridge)*(v @ v))/2
    scale = np.sum((D(box)*np.sum(np.abs(A), axis=1) + np.abs(b))**2)
    scale += D(ridge)*len(w)*D(box)**2
    margin = 32 * (sum(X.shape) + 4) * np.finfo(D).eps * scale
    lb = q - grad @ v - D(box)*np.sum(np.abs(grad)) - margin
    r64 = X @ w - y
    obj = float(0.5*(r64 @ r64 + ridge*(w @ w)))
    return _result(w, obj, min(obj, max(0.0, np.nextafter(float(lb), -np.inf))))


def _result(w, obj, lb):
    return {'weights': w.tolist(), 'objective': float(obj),
            'certified_lower_bound': float(lb),
            'certificate_leaves': [{'fixed_zero': [], 'fixed_nonzero': [],
                                    'leaf_lower_bound': float(lb)}]}
