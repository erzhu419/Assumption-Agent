"""Global box-constrained sparse regression with a disjunctive dual bound."""

import numpy as np
from scipy.optimize import lsq_linear


def solve_certified_box_l0(problem: dict) -> dict:
    """Return a feasible incumbent and an exhaustive, single-subcube certificate."""
    X = np.ascontiguousarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    tol = float(problem['gap_tol'])
    n, p = X.shape
    cutoff = 1e-10
    small_box = min(cutoff, box)

    def result(w, upper, lower):
        lower = float(min(lower, upper))
        return {'weights': w, 'objective': float(upper),
                'certified_lower_bound': lower,
                'certificate_leaves': [{'fixed_zero': [], 'fixed_nonzero': [],
                                        'leaf_lower_bound': lower}]}

    if p == 0:
        value = 0.5 * (y @ y)
        return result(np.zeros(0), value, value)

    count = 1 << p
    bits = 1 << np.arange(p)
    rows = np.arange(p)
    gram = X.T @ X
    gram[rows, rows] += ridge
    rhs = X.T @ y
    # A regularizer for initial guesses only, not for the objective or bound.
    jitter = 32 * np.finfo(float).eps * np.max(gram.diagonal())
    xnorm = np.linalg.norm(X, axis=0)
    ynorm = np.linalg.norm(y)
    rounding = 8 * np.finfo(float).eps * (n + 4 * p + 12)

    def evaluate(w, bounds, paid_count):
        r = np.einsum('ij,kj->ki', X, w) - y
        f = 0.5 * (np.einsum('ki,ki->k', r, r) + ridge * np.einsum('ki,ki->k', w, w))
        g = np.einsum('ij,ki->kj', X, r) + ridge * w
        # Retain the exact ridge curvature in the affine loss minorant. The
        # minimum over the box is coordinatewise, including zero ridge.
        if ridge:
            with np.errstate(over='ignore'):
                v = np.clip(w - g / ridge, -bounds, bounds)
        else:
            v = -np.sign(g) * bounds
        d = v - w
        lb = f + np.sum(d * (g + (0.5 * ridge) * d), axis=1) + penalty * paid_count
        # Allow for rounding of the residual, its dot products, and the sums.
        a = ynorm + np.abs(w) @ xnorm
        b = (np.abs(w) + bounds) @ xnorm
        scale = a * (a + b) + ridge * np.sum((np.abs(w) + bounds)**2, axis=1) + penalty * paid_count
        lb = np.maximum(lb - rounding * scale, 0.0)
        ub = f + penalty * np.count_nonzero(np.abs(w) > cutoff, axis=1)
        return lb, ub

    best = np.zeros(p)
    upper = float(0.5 * (y @ y))
    lower = np.empty(count)
    guesses = np.empty((count, p))
    # Bound temporary storage on tall inputs, while batching small systems.
    batch = max(1, min(512, 262144 // max(n, p * p)))
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        active = (np.arange(start, stop)[:, None] & bits) != 0
        k = active.sum(axis=1)
        bounds = np.where(active, box, small_box)
        mat = gram * active[:, :, None] * active[:, None, :]
        mat[:, rows, rows] += np.where(active, jitter, 1.0)
        if jitter:
            w = np.linalg.solve(mat, (rhs * active)[..., None])[..., 0]
            w = np.clip(w, -bounds, bounds)
        else:
            w = np.zeros(active.shape)
        guesses[start:stop] = w
        lower[start:stop], ub = evaluate(w, bounds, k)
        i = int(ub.argmin())
        if ub[i] < upper:
            best, upper = w[i].copy(), float(ub[i])

    # Initial guesses often suffice to discard most supports. Refine only
    # those whose current dual still leaves room below the incumbent.
    work = np.flatnonzero(lower < upper - 0.85 * tol)
    # Accelerated projected gradient supplies better dual centers in batches.
    # It need not converge: only the recomputed dual bound can discard a mask.
    if work.size >= 64:
        work = work[np.argsort(lower[work])]
        abs_gram = np.abs(gram)
        for start in range(0, work.size, batch):
            ids = work[start:start + batch]
            ids = ids[lower[ids] < upper - 0.85 * tol]
            if not ids.size:
                continue
            active = (ids[:, None] & bits) != 0
            bounds = np.where(active, box, small_box)
            k = active.sum(axis=1)
            # First optimize the paid coordinates; evaluate the dual over the
            # full asymmetric box, including the tiny unpenalized coordinates.
            limits = np.where(active, box, 0.0)
            L = np.max(np.einsum('ij,kj->ki', abs_gram, active) * active, axis=1)
            inv_L = (1.0 / np.where(L > 0, L, 1.0))[:, None]
            w = guesses[ids].copy()
            z = w.copy()
            momentum = 1.0
            for step in range(1, 97):
                grad = np.einsum('ij,kj->ki', gram, z) - rhs
                v = np.clip(z - grad * inv_L, -limits, limits)
                next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) * 0.5
                z = v + ((momentum - 1.0) / next_momentum) * (v - w)
                w = v
                momentum = next_momentum
                if step % 16 == 0:
                    val, ub = evaluate(w, bounds, k)
                    lower[ids] = np.maximum(lower[ids], val)
                    i = int(ub.argmin())
                    if ub[i] < upper:
                        best, upper = w[i].copy(), float(ub[i])
                    keep = lower[ids] < upper - 0.85 * tol
                    if not np.any(keep):
                        break
                    if not np.all(keep):
                        ids, bounds, k, limits, inv_L, w, z = (
                            a[keep] for a in (ids, bounds, k, limits, inv_L, w, z))
        work = np.flatnonzero(lower < upper - 0.85 * tol)
    if work.size:
        A = np.vstack((X, np.sqrt(ridge) * np.eye(p))) if ridge else X
        b = np.r_[y, np.zeros(p)] if ridge else y
        for mask in work[np.argsort(lower[work])]:
            if lower[mask] >= upper - 0.85 * tol:
                continue
            active = (mask & bits) != 0
            bounds = np.where(active, box, small_box)
            k = int(active.sum())
            w = np.zeros(p)
            if k:
                w[active] = np.clip(lsq_linear(A[:, active], b, bounds=(-box, box),
                                               method='bvls', tol=1e-13).x, -box, box)
            val, ub = evaluate(w[None, :], bounds[None, :], k)
            lower[mask] = max(lower[mask], val[0])
            if ub[0] < upper:
                best, upper = w.copy(), float(ub[0])
            # The literal objective does not charge coefficients <= cutoff.
            # If their possible contribution matters, optimize them as well.
            if not np.all(active) and lower[mask] < upper - 0.85 * tol:
                w = np.clip(lsq_linear(A, b, bounds=(-bounds, bounds),
                                       method='bvls', tol=1e-13).x, -bounds, bounds)
                val, ub = evaluate(w[None, :], bounds[None, :], k)
                lower[mask] = max(lower[mask], val[0])
                if ub[0] < upper:
                    best, upper = w.copy(), float(ub[0])

    r = X @ best - y
    upper = 0.5 * (r @ r + ridge * (best @ best)) + penalty * np.count_nonzero(np.abs(best) > cutoff)
    lb = float(lower.min())
    if upper - lb > tol:
        raise RuntimeError(f'Could not close the certificate gap: {upper - lb:g}')
    return result(best, upper, lb)
