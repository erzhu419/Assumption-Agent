"""Box-constrained L0/L2 regression and a partition certificate."""

import numpy as np
from scipy.optimize import lsq_linear, minimize


def solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float,
                         lam2: float, M: float, eps: float = 1e-6) -> dict:
    """Minimize sparse ridge loss and certify it to absolute tolerance eps."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    lam0, lam2, M, eps = map(float, (lam0, lam2, M, eps))
    n, p = X.shape
    if y.shape != (n,) or p > 16 or not np.isfinite([lam0, lam2, M, eps]).all():
        raise ValueError('invalid dimensions or parameters')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or min(lam0, lam2, M) < 0 or eps <= 0:
        raise ValueError('expected finite inputs, nonnegative penalties and box, and positive eps')
    all_bits = (1 << p) - 1
    bit = 1 << np.arange(p)
    XL, yl = X.astype(np.longdouble), y.astype(np.longdouble)
    L0, L2, box = map(np.longdouble, (lam0, lam2, M))
    abs_XL = np.abs(XL)
    abs_yl = np.abs(yl)
    dot_guard = 8 * (n + 2) * np.finfo(np.longdouble).eps
    sum_guard = 8 * (p + 4) * np.finfo(np.longdouble).eps
    G, c = X.T @ X, X.T @ y
    best = np.zeros(p)
    upper = 0.5 * float(yl @ yl)
    fitted = set()

    def offer(beta):
        nonlocal best, upper
        b = beta.astype(np.longdouble)
        r = yl - XL @ b
        f = float((r @ r + L2 * (b @ b)) / 2 + L0 * np.count_nonzero(b))
        if f < upper:
            best, upper = beta.copy(), f

    def ridge(mask):
        indices = np.flatnonzero(mask & bit)
        beta = np.zeros(p)
        if indices.size and M > 0:
            A = X[:, indices]
            rhs = y
            if lam2:
                A = np.vstack((A, np.sqrt(lam2) * np.eye(len(indices))))
                rhs = np.r_[y, np.zeros(len(indices))]
            b = np.linalg.lstsq(A, rhs, rcond=None)[0]
            if np.any(np.abs(b) > M):
                b = lsq_linear(A, rhs, bounds=(-M, M), method='bvls', tol=1e-14).x
            beta[indices] = np.clip(b, -M, M)
        if mask not in fitted:
            offer(beta)
            fitted.add(mask)
        return beta

    # Extract certified diagonal curvature when the Gram matrix permits it.
    # Gershgorin gives G - diag(d) >= 0 without inverting G or relying on
    # an approximate minimum eigenvalue.
    d = None
    if p and M:
        GL = XL.T @ XL
        gerr = dot_guard * (abs_XL.T @ abs_XL)
        off = np.abs(GL) + gerr
        np.fill_diagonal(off, 0)
        diagonal = GL.diagonal() - gerr.diagonal()
        candidate = diagonal - (1 + sum_guard) * off.sum(axis=1)
        candidate -= sum_guard * np.abs(diagonal)
        if np.all(candidate >= 0) and np.any(candidate > 0):
            d = candidate

    def bound_terms(z, curvature, zero, active):
        # Conjugate of curvature*b^2/2 on the coefficient box.
        b = np.minimum(np.divide(z, curvature, out=np.full(p, box),
                                 where=curvature > 0), box)
        gain = b * (z - curvature * b / 2)
        terms = L0 - gain
        fixed = (active & bit) != 0
        free = ((zero | active) & bit) == 0
        val = terms[fixed].sum() + np.minimum(0, terms[free]).sum()
        return val - sum_guard * (p * L0 + gain[fixed | free].sum())

    def dual(u, zero, active):
        # Fenchel inequality for the squared residual; minimize the remaining
        # separable terms over beta and the free indicators.
        u = np.asarray(u, dtype=np.longdouble)
        abs_u = np.abs(u)
        z = np.abs(XL.T @ u) + dot_guard * (abs_XL.T @ abs_u)
        quad = (u @ u) / 2
        safe = yl @ u - quad + bound_terms(z, L2, zero, active)
        safe -= (dot_guard + sum_guard) * (abs_yl @ abs_u + quad)
        return float(np.nextafter(float(safe), -np.inf))

    def diagonal_dual(beta, zero, active):
        # Tangent to the remaining PSD quadratic at beta, retaining the
        # separable loss curvature before taking the indicator relaxation.
        b = np.asarray(beta, dtype=np.longdouble)
        r = yl - XL @ b
        rerr = dot_guard * (abs_yl + abs_XL @ np.abs(b))
        g = XL.T @ r + d * b
        z = (np.abs(g) + (abs_XL.T @ rerr) * (1 + dot_guard)
             + (dot_guard + sum_guard) * (abs_XL.T @ np.abs(r) + np.abs(d * b)))
        quad = (r @ r) / 2
        center = (d * b) @ b / 2
        safe = yl @ r - quad + center + bound_terms(z, d + L2, zero, active)
        safe -= (dot_guard + sum_guard) * (abs_yl @ np.abs(r) + quad + center)
        safe -= (1 + dot_guard) * ((abs_yl + np.abs(r) + rerr / 2) @ rerr)
        return float(np.nextafter(float(safe), -np.inf))

    if M == 0 or p == 0:
        return {'beta': best, 'upper_bound': upper, 'tree_leaves': [
            {'fixed_zero': [], 'fixed_active': [], 'lower_bound': dual(y, 0, 0)}]}

    # Disjoint row blocks give independent losses. Convexify each small
    # block jointly over its indicators rather than one column at a time.
    # For a block, a common lower level is feasible in the disjunctive
    # dual iff it lies below every support's box-ridge dual value.
    present = X != 0
    components = []
    remaining = np.ones(p, dtype=bool)
    used_rows = np.zeros(n, dtype=bool)
    while remaining.any():
        columns = np.zeros(p, dtype=bool)
        columns[np.flatnonzero(remaining)[0]] = True
        while True:
            rows = present[:, columns].any(axis=1)
            expanded = columns | present[rows].any(axis=0)
            if np.array_equal(columns, expanded):
                break
            columns = expanded
        components.append((np.flatnonzero(columns), rows))
        remaining[columns] = False
        used_rows |= rows

    if len(components) > 1 and max(len(idx) for idx, _ in components) <= 8:
        candidate = np.zeros(p)
        u = np.where(used_rows, 0., y)
        bounds = [dual(u, all_bits, 0)]
        for idx, rows in components:
            local_best = np.longdouble(np.inf)
            local_lb = np.inf
            local_y = yl[rows]
            local_X = XL[rows][:, idx]
            for subset in range(1 << len(idx)):
                mask = sum(int(bit[j]) for k, j in enumerate(idx) if subset & (1 << k))
                beta = ridge(mask)[idx]
                b = beta.astype(np.longdouble)
                residual = local_y - local_X @ b
                score = (residual @ residual + L2 * (b @ b)) / 2 + L0 * np.count_nonzero(b)
                if score < local_best:
                    local_best = score
                    candidate[idx] = beta
                u = np.zeros(n, dtype=np.longdouble)
                u[rows] = residual
                local_lb = min(local_lb, dual(u, all_bits ^ mask, mask))
            bounds.append(local_lb)
        offer(candidate)
        bounds = np.asarray(bounds, dtype=np.longdouble)
        bound = float(np.nextafter(float(bounds.sum() - sum_guard * np.abs(bounds).sum()), -np.inf))
        if bound >= upper - eps:
            return {'beta': best, 'upper_bound': upper, 'tree_leaves': [
                {'fixed_zero': [], 'fixed_active': [], 'lower_bound': bound}]}

    # Convex envelope for a free indicator: linear in |b| up to t, then
    # lam0 + lam2*b**2/2. Split each sign into a low and a high segment.
    t = min(M, np.sqrt(lam0) * np.sqrt(2 / lam2)) if lam2 and lam0 else (M if lam0 else 0.)
    slope = lam0 / t + 0.5 * lam2 * t if t else 0.

    def relax(zero, active, start, target):
        free = np.flatnonzero(((zero | active) & bit) == 0)
        fixed = np.flatnonzero((active & bit) != 0)
        if free.size == 0:
            beta = ridge(active)
            u = y - X @ beta
            lb = dual(u, zero, active)
            if d is not None:
                lb = max(lb, diagonal_dual(beta, zero, active))
            return beta, u, lb
        u = y - X @ start
        lb = dual(u, zero, active)
        if d is not None:
            lb = max(lb, diagonal_dual(start, zero, active))
        if lb >= target:
            return start, u, lb
        j, sign, linear, diag = list(fixed), [1.] * len(fixed), [0.] * len(fixed), [lam2] * len(fixed)
        bounds = [(-M, M)] * len(fixed)
        x0 = list(start[fixed])
        if not lam0:
            j.extend(free)
            sign.extend([1.] * len(free))
            linear.extend([0.] * len(free))
            diag.extend([lam2] * len(free))
            bounds.extend([(-M, M)] * len(free))
            x0.extend(start[free])
        else:
            for s in (1., -1.):
                v = np.maximum(0., s * start[free])
                for cap, cost, curv, initial in ((t, slope, 0., np.minimum(t, v)),
                                                (M-t, lam2*t, lam2, np.maximum(0., v-t))):
                    if cap > 0:
                        j.extend(free)
                        sign.extend([s] * len(free))
                        linear.extend([cost] * len(free))
                        diag.extend([curv] * len(free))
                        bounds.extend([(0., cap)] * len(free))
                        x0.extend(initial)
        j, sign, diag, linear = map(np.asarray, (j, sign, diag, linear))
        H = G[np.ix_(j, j)] * sign[:, None] * sign
        H.flat[::len(j)+1] += diag
        q = linear - c[j] * sign

        def objective(v):
            hv = H @ v
            return (0.5 * hv + q) @ v, hv + q

        fit = minimize(objective, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 2.3e-16, 'gtol': 2e-12, 'maxiter': 700, 'maxls': 40})
        beta = np.clip(np.bincount(j, sign * fit.x, minlength=p), -M, M)
        u_fit = y - X @ beta
        lb_fit = dual(u_fit, zero, active)
        if d is not None:
            lb_fit = max(lb_fit, diagonal_dual(beta, zero, active))
        return (beta, u_fit, lb_fit) if lb_fit >= lb else (start, u, lb)

    if lam0 == 0:
        b = ridge(all_bits)
        bound = dual(y - X @ b, 0, 0)
        if bound >= upper - eps:
            return {'beta': best, 'upper_bound': upper, 'tree_leaves': [
                {'fixed_zero': [], 'fixed_active': [], 'lower_bound': bound}]}

    def round_support(beta, free, active):
        # Evaluate actual box-ridge objectives, not the relaxation's objective.
        for frac in (0.5, 0.15, 0.85):
            select = free[(np.abs(beta[free]) >= frac * t) & (np.abs(beta[free]) > 0)]
            mask = active | int(bit[select].sum())
            if mask not in fitted:
                ridge(mask)

    search_tol = min(eps * 0.05, 2e-10)

    def search(zero, active, start):
        beta, u, lb = relax(zero, active, start, upper - search_tol)
        if lb >= upper - search_tol or (zero | active) == all_bits:
            return (zero, active, lb, beta, u, None)
        free = np.flatnonzero(((zero | active) & bit) == 0)
        round_support(beta, free, active)
        if lb >= upper - search_tol:
            return (zero, active, lb, beta, u, None)
        fraction = np.minimum(1., np.abs(beta[free]) / t) if t else np.zeros(len(free))
        i = int(np.argmax(np.minimum(fraction, 1 - fraction)))
        b = int(bit[free[i]])
        left = (zero | b, active)
        right = (zero, active | b)
        if fraction[i] >= 0.5:
            right, left = left, right
        children = (search(*left, beta), search(*right, beta))
        return zero, active, lb, beta, u, children

    if d is not None:
        # The diagonal model proposes an incumbent; ridge refits its support.
        curvature = G.diagonal() + lam2
        magnitude = np.minimum(np.divide(np.abs(c), curvature,
                                        out=np.full(p, M), where=curvature > 0), M)
        gain = magnitude * (np.abs(c) - curvature * magnitude / 2)
        ridge(int(bit[gain > lam0].sum()))

    tree = search(0, 0, best)

    target = upper - eps
    visited = {}
    partition = {}

    def compact(node):
        zero, active, lb, beta, u, children = node
        visited[zero, active] = (lb, beta, u)
        if lb >= target:
            partition[zero, active] = (lb, beta, u)
        elif children is None:
            raise ArithmeticError('could not certify a terminal box-ridge problem at the requested eps')
        else:
            compact(children[0])
            compact(children[1])

    compact(tree)
    # Variable choices can differ across branches. After collapsing the search
    # tree, also test sibling pairs that arise across its original branches.
    pending = list(partition)
    while pending:
        zero, active = key = pending.pop()
        if key not in partition:
            continue
        remaining = zero | active
        while remaining:
            b = remaining & -remaining
            remaining ^= b
            sibling = (zero ^ b, active ^ b)
            if sibling not in partition:
                continue
            parent = (zero & ~b, active & ~b)
            if parent not in visited:
                _, beta, u = partition[key]
                lb = dual(u, *parent)
                if lb < target:
                    _, other_beta, other_u = partition[sibling]
                    other_lb = dual(other_u, *parent)
                    if other_lb > lb:
                        lb, beta, u = other_lb, other_beta, other_u
                if lb < target:
                    beta, u, lb = relax(*parent, beta, target)
                visited[parent] = (lb, beta, u)
            info = visited[parent]
            if info[0] >= target:
                del partition[key], partition[sibling]
                partition[parent] = info
                pending.append(parent)
                break

    leaves = [{'fixed_zero': np.flatnonzero(zero & bit).tolist(),
               'fixed_active': np.flatnonzero(active & bit).tolist(),
               'lower_bound': info[0]}
              for (zero, active), info in partition.items()]
    if len(leaves) > 90:
        # Strengthen the root bound using a joint disjunction over supports.
        # Each support has its own box-ridge residual dual. In the lifted
        # convex hull, support weights are nonnegative and sum to one, so
        # their weighted dual values are bounded below by their minimum.
        # With p <= 16 this costs at most 65,536 small box-ridge solves and
        # is only needed when the cheap perspective tree misses the cap.
        root_lb = np.inf
        for mask in range(1 << p):
            beta = ridge(mask)
            u = yl - XL @ beta.astype(np.longdouble)
            root_lb = min(root_lb, dual(u, all_bits ^ mask, mask))
        if root_lb < upper - eps:
            raise ArithmeticError('could not certify the joint box-ridge bound at the requested eps')
        leaves = [{'fixed_zero': [], 'fixed_active': [], 'lower_bound': root_lb}]
    return {'beta': best, 'upper_bound': upper, 'tree_leaves': leaves}
