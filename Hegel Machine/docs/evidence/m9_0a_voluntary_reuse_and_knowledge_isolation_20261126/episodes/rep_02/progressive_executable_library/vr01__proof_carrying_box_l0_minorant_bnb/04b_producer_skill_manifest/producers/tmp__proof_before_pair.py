"""Branch-and-bound with diagonal-curvature dual certificates."""

def __hegel_load_skill_autogenous_prim_box_ridge_bounds():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def gram_diagonal_minorant(design: 'np.ndarray') -> 'np.ndarray':
        """Return a nonnegative vector d (p,) for design X (n, p) such that X.T @ X - diag(d) is positive semidefinite."""
        x = np.asarray(design, dtype=np.longdouble)
        (n, p) = x.shape
        eps = np.finfo(x.dtype).eps
        dot_guard = 8 * (n + 2) * eps
        sum_guard = 8 * (p + 2) * eps
        mag = np.abs(x)
        gram = x.T @ x
        err = dot_guard * (mag.T @ mag)
        off = np.abs(gram) + err
        np.fill_diagonal(off, 0)
        diag_lo = gram.diagonal() - err.diagonal()
        d = diag_lo - (1 + sum_guard) * off.sum(axis=1) - sum_guard * np.abs(diag_lo)
        return d if np.all(d >= 0) else np.zeros(p, dtype=x.dtype)
    return __hegel_types.SimpleNamespace(gram_diagonal_minorant=gram_diagonal_minorant)
__hegel_skill_ns_autogenous_prim_box_ridge_bounds = __hegel_load_skill_autogenous_prim_box_ridge_bounds()
import numpy as np
from scipy.optimize import minimize
gram_diagonal_minorant = __hegel_skill_ns_autogenous_prim_box_ridge_bounds.gram_diagonal_minorant

def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    (l0, l2, M) = (float(problem[k]) for k in ('lam0', 'lam2', 'M'))
    (_, p) = X.shape
    G = X.T @ X
    xy = X.T @ y
    d = np.asarray(gram_diagonal_minorant(X), dtype=float)
    R = G - np.diag(d)
    scale = max(1.0, np.linalg.norm(G, np.inf))
    if np.linalg.eigvalsh(R)[0] < 1e-12 * scale:
        d = d * (1 - 1e-10)
        R = G - np.diag(d)
        if np.linalg.eigvalsh(R)[0] < 0.0:
            d = np.zeros(p)
            R = G.copy()
    a = d + l2
    H = G + l2 * np.eye(p)
    diagR = np.diag(R)
    diagH = np.diag(H)
    t = np.minimum(M, np.sqrt(2 * l0 / a))
    slope = l0 / t + 0.5 * a * t
    best = np.zeros(p)
    bestval = 0.5 * (y @ y)

    def incumbent(support, start):
        nonlocal best, bestval
        idx = np.flatnonzero(support)
        if not len(idx):
            return
        hh = H[np.ix_(idx, idx)]
        h = xy[idx]
        v = np.linalg.solve(hh, h)
        if np.max(np.abs(v)) > M:

            def fun(v):
                hv = hh @ v
                return ((0.5 * hv - h) @ v, hv - h)
            v = minimize(fun, np.clip(start[idx], -M, M), jac=True, method='L-BFGS-B', bounds=[(-M, M)] * len(idx), options={'gtol': 1e-11, 'ftol': 1e-15}).x
            free = np.abs(v) < M
            trial = v.copy()
            trial[free] = np.linalg.solve(hh[np.ix_(free, free)], h[free] - hh[np.ix_(free, ~free)] @ v[~free])
            if np.max(np.abs(trial)) <= M and fun(trial)[0] < fun(v)[0]:
                v = trial
        w = np.zeros(p)
        w[idx] = np.where(np.abs(v) > 1e-08, v, 0.0)
        r = y - X @ w
        f = 0.5 * (r @ r + l2 * (w @ w)) + l0 * np.count_nonzero(w)
        if f < bestval:
            (best, bestval) = (w, float(f))

    def relax(z, on, start):
        free = ~(z | on)
        x = start.copy()
        x[z] = 0.0
        g = R @ x - xy
        indices = np.flatnonzero(~z)
        for sweep in range(240):
            delta = 0.0
            for j in indices:
                h = diagR[j] * x[j] - g[j]
                mag = abs(h)
                if on[j] or mag >= diagR[j] * t[j] + slope[j]:
                    v = min(M, mag / diagH[j])
                elif mag <= slope[j]:
                    v = 0.0
                else:
                    v = min(t[j], (mag - slope[j]) / diagR[j])
                change = np.copysign(v, h) - x[j]
                if change:
                    x[j] += change
                    g += change * R[:, j]
                    delta = max(delta, abs(change))
            if delta <= 2e-10 * max(1.0, M):
                break
        r = y - X @ x
        corr = np.abs(X.T @ r + d * x)
        b = np.minimum(M, corr / a)
        cost = l0 - b * (corr - 0.5 * a * b)
        lb = 0.5 * (r @ r) + r @ (X @ x) + 0.5 * (d * x @ x)
        lb += np.sum(cost[on]) + np.sum(np.minimum(0.0, cost[free]))
        lb -= 2e-12 * (1 + abs(lb) + y @ y)
        return (x, float(lb), cost)
    blank = np.zeros(p, dtype=bool)
    incumbent(np.abs(xy) > slope, best)
    leaves = []
    stack = [(blank, blank, best)]
    while stack:
        (z, on, start) = stack.pop()
        (x, lb, cost) = relax(z, on, start)
        free = ~(z | on)
        if lb < bestval - 4e-06:
            incumbent(on | free & (np.abs(x) >= 0.5 * t), x)
            if lb < bestval - 4e-06:
                incumbent(on | free & (np.abs(x) >= t * (1 - 1e-07)), x)
        if lb >= bestval - 4e-06:
            leaves.append({'fixed_zero': z, 'fixed_active': on, 'lb': lb, 'minorant_diagonal': d, 'minorant_center': x})
            continue
        if not free.any():
            raise RuntimeError('Could not close a fixed-support dual gap')
        fraction = np.minimum(np.abs(x) / t, 1.0)
        score = fraction * (1 - fraction)
        score[~free] = -1.0
        j = int(np.argmax(score))
        if score[j] < 1e-10:
            j = int(np.argmin(np.where(free, np.abs(cost), np.inf)))
        (zz, oo) = (z.copy(), on.copy())
        (zz[j], oo[j]) = (True, True)
        (zero, active) = ((zz, on, x), (z, oo, x))
        if fraction[j] >= 0.5:
            stack.extend((zero, active))
        else:
            stack.extend((active, zero))
    if len(leaves) > 2048:
        raise RuntimeError('Certificate exceeds the leaf budget')
    return {'weights': best, 'objective': bestval, 'certified_lower_bound': min(bestval, min((q['lb'] for q in leaves))), 'subcube_leaves': leaves}
