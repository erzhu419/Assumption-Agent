"""Small-dimensional box-constrained L0 regression with dual subcube bounds."""

from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import lsq_linear


_NONZERO = 1e-10


def solve_certified_box_l0(problem: dict) -> dict:
    """Enumerate binary supports, bound them by box-QP duality, then group them.

    A support pays for its selected coordinates even if a box-QP minimizer
    sets some of them to zero. For the numerical L0 convention, unselected
    coordinates may lie in the tiny zero-count box. Relaxing them this way
    also gives a lower bound for subproblems that fix them to exactly zero.
    """
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    n, p = X.shape
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    tol = float(problem['gap_tol'])
    max_leaves = int(problem['max_leaves'])
    small_box = min(box, _NONZERO)
    G = X.T @ X + ridge * np.eye(p)
    h = X.T @ y

    # A well-conditioned proposal is useful, but never needed for validity
    # of the dual bound. This shift is not part of the objective or certificate.
    shift = 32 * np.finfo(float).eps * max(1., np.max(np.abs(G), initial=0.)) * max(1, p)
    proposal_G = G + shift * np.eye(p)

    ld = np.longdouble
    Xl, yl = X.astype(ld), y.astype(ld)
    yabs = np.abs(yl)
    row_l1 = np.sum(np.abs(Xl), axis=1)
    guard = 16 * np.finfo(ld).eps * (n + p + 4)
    bits = 1 << np.arange(p)
    all_ids = np.arange(1 << p)
    lower = np.empty(1 << p)
    counts = np.zeros(1 << p, dtype=np.int8)
    best_w = np.zeros(p)
    best = float(0.5 * (y @ y))

    def dual(w, radii, k):
        # r is an arbitrary residual-space dual vector. The ridge tangent
        # at w and the minimum of a linear form over the box give
        # -||r||²/2 - y'r - ridge*||w||²/2 - radii'|X'r+ridge*w| + k*penalty.
        r = (np.einsum('ij,bj->bi', X, w) - y).astype(ld)
        wl = w.astype(ld)
        rr = np.sum(r * r, axis=1) / 2
        yr = np.einsum('bi,i->b', r, yl)
        ww = ld(ridge) * np.sum(wl * wl, axis=1) / 2
        grad = np.einsum('bi,ij->bj', r, Xl) + ld(ridge) * wl
        spread = np.sum(radii * np.abs(grad), axis=1)
        charge = ld(penalty) * k
        # Accumulation error uses absolute terms before cancellation.
        scale = (rr + np.einsum('bi,i->b', np.abs(r), yabs) + ww + charge
                 + box * (np.einsum('bi,i->b', np.abs(r), row_l1)
                          + ridge * np.sum(np.abs(wl), axis=1)))
        value = charge - rr - yr - ww - spread - guard * scale
        return np.maximum(0., np.nextafter(value.astype(float), -np.inf))

    def update(w):
        nonlocal best, best_w
        r = np.einsum('ij,bj->bi', X, w) - y
        v = (0.5 * np.sum(r*r, axis=1) + 0.5 * ridge * np.sum(w*w, axis=1)
             + penalty * np.count_nonzero(np.abs(w) > _NONZERO, axis=1))
        i = np.argmin(v)
        if v[i] < best:
            best, best_w = float(v[i]), w[i].copy()

    # Group by cardinality for batched unconstrained linear solves. Both the
    # unconstrained proposal and its box projection supply valid dual vectors.
    zero = np.zeros((1, p))
    lower[0] = dual(zero, np.full((1, p), small_box), 0)[0]
    for k in range(1, p + 1):
        idx = np.fromiter((j for group in combinations(range(p), k) for j in group),
                          dtype=np.intp, count=k * comb(p, k)).reshape(-1, k)
        masks = np.sum(bits[idx], axis=1)
        counts[masks] = k
        charge = np.nextafter(k * penalty, -np.inf)
        if charge >= best:
            lower[masks] = charge
            continue
        for start in range(0, len(idx), 512):
            j = idx[start:start+512]
            ids = masks[start:start+512]
            rows = np.arange(len(j))[:, None]
            a = proposal_G[j[:, :, None], j[:, None, :]]
            u = np.linalg.solve(a, h[j][..., None])[..., 0]
            radii = np.full((len(j), p), small_box)
            radii[rows, j] = box
            w = np.zeros_like(radii)
            w[rows, j] = np.clip(u, -box, box)
            update(w)
            lb = dual(w, radii, k)
            outside = np.flatnonzero(np.any(np.abs(u) > box, axis=1) & (lb < best))
            if len(outside):
                # Limit very large proposals from nearly singular systems;
                # the dual remains valid for the limited vector as well.
                wp = w[outside].copy()
                wp[np.arange(len(outside))[:, None], j[outside]] = np.clip(u[outside], -32*box, 32*box)
                lb[outside] = np.maximum(lb[outside], dual(wp, radii[outside], k))
            lower[ids] = np.maximum(lb, charge)

    target_gap = 0.75 * tol
    pending = np.flatnonzero(lower < best - target_gap)
    # When many supports have weak initial bounds, batched projected
    # coordinate sweeps improve the duals before individual BVLS calls.
    if len(pending) > 48:
        for start in range(0, len(pending), 512):
            ids = pending[start:start+512]
            ids = ids[lower[ids] < best - target_gap]
            if not len(ids):
                continue
            radii = np.where(ids[:, None] & bits, box, small_box)
            w = np.clip(np.broadcast_to(best_w, radii.shape), -radii, radii)
            g = np.einsum('bi,ij->bj', w, G) - h
            for _ in range(24):
                for j in range(p):
                    if G[j, j] > 0.:
                        new = np.clip(w[:, j] - g[:, j] / G[j, j], -radii[:, j], radii[:, j])
                        g += (new - w[:, j])[:, None] * G[j]
                        w[:, j] = new
            update(w)
            lower[ids] = np.maximum(lower[ids], dual(w, radii, counts[ids]))

    pending = np.flatnonzero(lower < best - target_gap)
    if len(pending):
        if ridge:
            A = np.vstack((X, np.sqrt(ridge) * np.eye(p)))
            b = np.r_[y, np.zeros(p)]
        else:
            A, b = X, y
        for mask in pending[np.argsort(lower[pending])]:
            if lower[mask] >= best - target_gap:
                continue
            radii = np.where(mask & bits, box, small_box)
            fit = lsq_linear(A, b, bounds=(-radii, radii), method='bvls', tol=1e-13)
            w = np.clip(fit.x, -radii, radii)[None, :]
            update(w)
            lower[mask] = max(lower[mask], dual(w, radii[None, :], counts[mask])[0])

    # Collapse supports into sibling subcubes along the incumbent's path.
    # Each leaf inherits the minimum of the individual support dual bounds.
    active = np.abs(best_w) > _NONZERO
    order = list(np.flatnonzero(active)) + list(np.flatnonzero(~active))
    leaves = []

    def add_leaf(ids, zero, nonzero):
        leaves.append({'fixed_zero': list(map(int, zero)),
                       'fixed_nonzero': list(map(int, nonzero)),
                       'leaf_lower_bound': float(np.min(lower[ids]))})

    prefix_zero, prefix_nonzero = [], []
    for j in order[:max_leaves - 1]:
        on = (all_ids & bits[j]) != 0
        if active[j]:
            add_leaf(all_ids[~on], prefix_zero + [j], prefix_nonzero)
            all_ids = all_ids[on]
            prefix_nonzero.append(j)
        else:
            add_leaf(all_ids[on], prefix_zero, prefix_nonzero + [j])
            all_ids = all_ids[~on]
            prefix_zero.append(j)
    add_leaf(all_ids, prefix_zero, prefix_nonzero)

    r = X @ best_w - y
    objective = float(0.5 * (r @ r) + 0.5 * ridge * (best_w @ best_w)
                      + penalty * np.count_nonzero(active))
    bound = min(objective, min(leaf['leaf_lower_bound'] for leaf in leaves))
    if objective - bound > tol:
        pass
        ids = np.argsort(lower)[:8]; print('DIAGNOSTIC', objective, bound, ids, lower[ids], 'incumbent', np.sum(bits[active]))
    return {'weights': best_w.tolist(), 'objective': objective,
            'certified_lower_bound': bound, 'certificate_leaves': leaves}
