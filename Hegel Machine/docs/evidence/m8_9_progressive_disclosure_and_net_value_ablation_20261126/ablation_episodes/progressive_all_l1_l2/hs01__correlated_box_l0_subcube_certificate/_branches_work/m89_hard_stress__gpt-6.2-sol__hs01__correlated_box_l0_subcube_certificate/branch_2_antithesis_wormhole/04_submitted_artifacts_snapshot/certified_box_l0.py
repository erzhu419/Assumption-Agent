"""Box-constrained sparse ridge with a disjunctive dual certificate."""
from itertools import combinations

import numpy as np
from scipy.optimize import lsq_linear


def solve_certified_box_l0(problem: dict) -> dict:
    """Minimize the thresholded L0 objective and certify the entire cube."""
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    tol = float(problem['gap_tol'])
    n, p = X.shape
    tiny = min(box, 1e-10)
    H = X.T @ X + ridge * np.eye(p)
    c = X.T @ y
    # Regularize only the numerical seed, never the problem or its bounds.
    jitter = 32 * np.finfo(float).eps * max(1., np.max(np.abs(H), initial=0.))
    A = X.astype(np.longdouble)
    a = np.abs(A)
    h = y.astype(np.longdouble)
    guard = 16 * (max(n, p) + 4) * np.finfo(np.longdouble).eps
    best = np.zeros(p)
    upper = float(0.5 * (y @ y))

    def dual_and_incumbent(w, active):
        nonlocal best, upper
        r = y - np.einsum('ij,bj->bi', X, w)
        obj = (0.5 * np.einsum('bi,bi->b', r, r)
               + 0.5 * ridge * np.einsum('bi,bi->b', w, w)
               + penalty * np.count_nonzero(np.abs(w) > 1e-10, axis=1))
        j = np.argmin(obj)
        if obj[j] < upper:
            best = w[j].copy()
            rr = y - X @ best
            upper = float(0.5 * (rr @ rr) + 0.5 * ridge * (best @ best)
                          + penalty * np.count_nonzero(np.abs(best) > 1e-10))
        # Any residual is a valid Fenchel dual variable. Inactive coordinates
        # may move up to the counting threshold, so the bound also applies
        # to the stated thresholded objective, not just exact sparsity.
        d = r.astype(np.longdouble)
        z = np.abs(np.einsum('bi,ij->bj', d, A))
        limits = np.where(active, box, tiny).astype(np.longdouble)
        t = np.minimum(limits, z / ridge) if ridge else limits
        conj = t * (z - (np.longdouble(ridge) / 2) * t)
        linear = d * h
        sq = 0.5 * d * d
        cost = np.longdouble(penalty) * active.sum(axis=1)
        bound = linear.sum(axis=1) - sq.sum(axis=1) + cost - conj.sum(axis=1)
        # Bound dot-product error through the conjugate's box Lipschitz
        # constant, and round the final scalar toward minus infinity.
        scale = (np.abs(linear).sum(axis=1) + sq.sum(axis=1) + cost
                 + (limits * np.einsum('bi,ij->bj', np.abs(d), a)).sum(axis=1)
                 + (np.longdouble(ridge) / 2) * (t*t).sum(axis=1))
        return np.nextafter(np.asarray(bound - guard * scale, dtype=float), -np.inf)

    def mask_for(ids):
        active = np.zeros((len(ids), p), dtype=bool)
        active[np.arange(len(ids))[:, None], ids] = True
        return active

    def pack(ids, v):
        w = np.zeros((len(ids), p))
        w[np.arange(len(ids))[:, None], ids] = v
        return w

    def solve_batch(G, rhs):
        return np.linalg.solve(G, rhs[..., None])[..., 0]

    pending = []
    floor = np.inf
    margin = 0.9 * tol
    for k in range(p + 1):
        ids = np.array(list(combinations(range(p), k)), dtype=np.intp).reshape(-1, k) if k else np.empty((1, 0), dtype=np.intp)
        for ii in range(0, len(ids), 1024):
            idx = ids[ii:ii+1024]
            active = mask_for(idx)
            if k:
                G = H[idx[:, :, None], idx[:, None, :]]
                G[:, np.arange(k), np.arange(k)] += jitter
                v = np.clip(solve_batch(G, c[idx]), -box, box)
            else:
                v = np.empty((1, 0))
            bounds = dual_and_incumbent(pack(idx, v), active)
            keep = bounds < upper - margin
            if np.any(keep):
                pending.append((idx[keep], v[keep], bounds[keep]))
            floor = min(floor, np.min(bounds[~keep], initial=np.inf))

    # Improve the primal early on the strongest numerical candidates.
    # BVLS is also the rank-safe fallback if a batch active set has not settled.
    aug_y = np.r_[y, np.zeros(p)]
    aug_X = np.vstack((X, np.sqrt(ridge) * np.eye(p)))

    def refine(idx):
        active = mask_for(idx[None, :])
        limits = np.where(active[0], box, tiny)
        if not p:
            return dual_and_incumbent(np.zeros((1, 0)), active)[0]
        fit = lsq_linear(aug_X, aug_y, bounds=(-limits, limits), method='bvls', tol=1e-13)
        w = np.clip(fit.x, -limits, limits)[None, :]
        return dual_and_incumbent(w, active)[0]

    starts = sorted(((float(b.min()), j, int(b.argmin())) for j, (_, _, b) in enumerate(pending)))[:12]
    for _, j, i in starts:
        idx, _, bounds = pending[j]
        if bounds[i] < upper - margin:
            bounds[i] = max(bounds[i], refine(idx[i]))

    unresolved = []
    for idx, v, bounds in pending:
        k = idx.shape[1]
        for _ in range(6):
            keep = bounds < upper - margin
            floor = min(floor, np.min(bounds[~keep], initial=np.inf))
            idx, v, bounds = idx[keep], v[keep], bounds[keep]
            if not len(idx) or not k:
                break
            # Primal-dual active-set step. Fix predicted bound coordinates,
            # solve only the free equations; clipped results are feasible even
            # if the predicted active set is wrong.
            G = H[idx[:, :, None], idx[:, None, :]]
            q = c[idx]
            grad = np.einsum('bij,bj->bi', G, v) - q
            diag = np.diagonal(G, axis1=1, axis2=2)
            pred = v - grad / np.maximum(diag, jitter)
            free = np.abs(pred) < box
            fixed = np.where(free, 0., np.copysign(box, pred))
            rhs = np.where(free, q - np.einsum('bij,bj->bi', G, fixed), fixed)
            K = G * free[:, :, None] * free[:, None, :]
            K[:, np.arange(k), np.arange(k)] += np.where(free, jitter, 1.)
            new = np.clip(solve_batch(K, rhs), -box, box)
            bounds = np.maximum(bounds, dual_and_incumbent(pack(idx, new), mask_for(idx)))
            if np.array_equal(new, v):
                break
            v = new
        if len(idx):
            unresolved.append((idx, bounds))

    for idx, bounds in unresolved:
        for row, lb in zip(idx, bounds):
            if lb < upper - margin:
                lb = max(lb, refine(row))
            floor = min(floor, lb)

    lower = min(upper, float(floor))
    if upper - lower > tol:
        raise ArithmeticError('could not close the box L0 dual gap')
    # The root is the union of all fully specified supports; the minimum
    # of the support dual bounds therefore certifies one exhaustive leaf.
    return {'weights': best, 'objective': upper, 'certified_lower_bound': lower,
            'certificate_leaves': [{'fixed_zero': [], 'fixed_nonzero': [],
                                    'leaf_lower_bound': lower}]}
