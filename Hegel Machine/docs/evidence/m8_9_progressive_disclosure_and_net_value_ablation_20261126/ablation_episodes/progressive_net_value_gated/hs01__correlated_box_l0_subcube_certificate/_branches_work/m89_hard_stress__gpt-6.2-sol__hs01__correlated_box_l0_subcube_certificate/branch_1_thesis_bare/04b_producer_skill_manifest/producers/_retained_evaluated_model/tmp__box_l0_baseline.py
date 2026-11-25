"""Box-constrained L0 optimization with a support-enumeration certificate."""

from itertools import combinations

import numpy as np
from scipy.optimize import lsq_linear


_THRESHOLD = 1e-10


def solve_certified_box_l0(problem: dict) -> dict:
    """Return a feasible incumbent and an exhaustive, single-subcube certificate.

    For each support mask we bound the associated box QP by weak duality.
    The minimum of those bounds certifies the root cube; the leaf limit
    constrains the reported partition, not the number of QPs considered.
    """
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    n, p = X.shape
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    gap = float(problem['gap_tol'])
    small = min(box, _THRESHOLD)
    eps = np.finfo(float).eps
    absX = np.abs(X)
    abs_y = np.abs(y)
    cols = np.arange(p)
    bits = 1 << cols
    gram = X.T @ X + ridge * np.eye(p)
    cross = X.T @ y

    def score(w):
        r = X @ w - y
        return float(0.5 * (r @ r + ridge * (w @ w))
                     + penalty * np.count_nonzero(np.abs(w) > _THRESHOLD))

    def evaluate(w, widths, cost):
        # For any a, ||Xv-y||^2/2 >= a^T(Xv-y) - ||a||^2/2.
        # Minimize the resulting separable ridge-plus-linear function on
        # this mask's box, including tiny unpenalized coefficients.
        a = np.einsum('bp,np->bn', w, X) - y
        g = np.einsum('bn,np->bp', a, X)
        if ridge:
            # On the interior of the one-dimensional box minimum the
            # minimizer is -g/ridge; elsewhere it is the appropriate end.
            inside = np.abs(g) < ridge * widths
            z = -np.copysign(widths, g)
            np.divide(-g, ridge, out=z, where=inside)
            terms = (g + 0.5 * ridge * z) * z
        else:
            terms = -np.abs(g) * widths
        aa = 0.5 * np.einsum('bn,bn->b', a, a)
        const = -np.einsum('bn,n->b', a, y) - aa
        dual = const + terms.sum(axis=1) + cost
        # Roundoff allowance scales with absolute products, not with the
        # (possibly heavily cancelled) value of the dual objective.
        gx = np.einsum('bn,np->bp', np.abs(a), absX)
        mag = (np.einsum('bn,n->b', np.abs(a), abs_y) + aa
               + np.sum(widths * (gx + 0.5 * ridge * widths), axis=1) + cost)
        dual -= eps * (8 * max(n, p) + 32) * mag
        primal = (aa + 0.5 * ridge * np.einsum('bp,bp->b', w, w)
                  + penalty * np.count_nonzero(np.abs(w) > _THRESHOLD, axis=1))
        return dual, primal

    best_w = np.zeros(p)
    best = score(best_w)
    bounds = np.empty(1 << p)
    if p:
        # This shift is only for trial points (especially on singular
        # supports); no Gram-matrix inverse is used in the certificate.
        shift = eps * (32 * p) * float(np.max(np.diag(gram)))
        trial_gram = gram + max(shift, np.finfo(float).tiny) * np.eye(p)
    for k in range(p + 1):
        supports = np.array(list(combinations(range(p), k)), dtype=int)
        for start in range(0, len(supports), 512):
            I = supports[start:start + 512]
            ids = bits[I].sum(axis=1)
            w = np.zeros((len(I), p))
            widths = np.full_like(w, small)
            rows = np.arange(len(I))[:, None]
            if k:
                w[rows, I] = np.linalg.solve(
                    trial_gram[I[:, :, None], I[:, None, :]], cross[I, None]
                )[..., 0]
                widths[rows, I] = box
            cost = k * penalty
            unclipped = np.any(np.abs(w) > widths, axis=1)
            if np.any(unclipped):
                outside_dual, _ = evaluate(w[unclipped], widths[unclipped], cost)
            np.clip(w, -widths, widths, out=w)
            dual, primal = evaluate(w, widths, cost)
            if np.any(unclipped):
                dual[unclipped] = np.maximum(dual[unclipped], outside_dual)
            bounds[ids] = dual
            j = int(np.argmin(primal))
            if primal[j] < best:
                v = score(w[j])
                if v < best:
                    best_w, best = w[j].copy(), v

    # Refine only supports whose dual bounds can still limit the gap.
    if bounds.min() < best - 0.9 * gap:
        if ridge:
            A = np.vstack((X, np.sqrt(ridge) * np.eye(p)))
            b = np.concatenate((y, np.zeros(p)))
        else:
            A, b = X, y
        for mask in np.argsort(bounds):
            if bounds[mask] >= best - 0.9 * gap:
                break
            active = (mask & bits) != 0
            widths = np.where(active, box, small)
            result = lsq_linear(A, b, bounds=(-widths, widths), method='bvls', tol=1e-13)
            w = np.clip(result.x, -widths, widths)
            dual, _ = evaluate(w[None, :], widths[None, :], int(active.sum()) * penalty)
            bounds[mask] = max(bounds[mask], dual[0])
            v = score(w)
            if v < best:
                best_w, best = w, v

    lower = min(best, float(bounds.min()))
    if best - lower > gap:
        raise RuntimeError('Could not close the certified box-L0 gap at float64 precision')
    return {
        'weights': best_w.tolist(),
        'objective': best,
        'certified_lower_bound': lower,
        'certificate_leaves': [
            {'fixed_zero': [], 'fixed_nonzero': [], 'leaf_lower_bound': lower}
        ],
    }
