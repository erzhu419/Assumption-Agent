"""Branch-and-bound for box-constrained sparse ridge regression."""

import heapq
import itertools
import math

import numpy as np
from scipy.optimize import minimize


_THRESHOLD = 1e-9


def _minorant(H, ridge):
    """Choose a diagonal curvature split, then verify it by factorization."""
    p = len(H)
    scale = np.sqrt(H.diagonal())
    R = H / scale / scale[:, None]
    rho = max(0., np.linalg.eigvalsh(R)[0])
    sizes = [np.full(p, max(ridge, np.linalg.eigvalsh(H)[0])),
             np.maximum(ridge, rho * H.diagonal())]
    for d in sorted(sizes, key=np.sum, reverse=True):
        d = ridge + .985 * (d - ridge)
        K = H - np.diag(d)
        slack = 64 * np.finfo(float).eps * (1 + np.max(np.abs(H))) * p
        if np.min(K.diagonal()) <= slack:
            continue
        try:
            L = np.linalg.cholesky(K - slack * np.eye(p))
        except np.linalg.LinAlgError:
            continue
        if np.max(np.sum(np.abs(K - slack * np.eye(p) - L @ L.T), axis=1)) < .25 * slack:
            return d
    return np.full(p, ridge)


def solve_certified_box_l0(instance: dict) -> dict:
    X = np.asarray(instance['X'], dtype=float)
    y = np.asarray(instance['y'], dtype=float)
    ridge = float(instance['ridge'])
    M = float(instance['box'])
    fee = float(instance['l0_penalty'])
    p = X.shape[1]
    # A row couples precisely the columns that are nonzero in that row.
    parent = list(range(p))

    def root(j):
        while parent[j] != j:
            j = parent[j]
        return j

    incidence = X != 0
    for row in incidence:
        cols = np.flatnonzero(row)
        if len(cols) > 1:
            first = root(cols[0])
            for j in cols[1:]:
                parent[root(j)] = first
    groups = {}
    for j in range(p):
        groups.setdefault(root(j), []).append(j)
    if len(groups) <= 1:
        result = _gram_blocks(X, y, ridge, M, fee)
        return result if result is not None else _solve_block(X, y, ridge, M, fee, 600)

    weights = np.zeros(p)
    idle = y[~np.any(incidence, axis=1)]
    idle_loss = .5 * (idle @ idle)
    bounds = [idle_loss - 8 * np.finfo(float).eps * max(1, len(idle)) * idle_loss]
    nodes = 0
    for i, cols in enumerate(groups.values()):
        rows = np.any(incidence[:, cols], axis=1)
        budget = 600 - nodes - (len(groups) - 1 - i)
        part = _solve_block(X[rows][:, cols], y[rows], ridge, M, fee, budget)
        weights[cols] = part['weights']
        nodes += part['nodes_explored']
        bounds.append(part['certified_lower_bound'])
    residual = X @ weights - y
    value = .5 * (residual @ residual + ridge * (weights @ weights))
    value += fee * np.count_nonzero(np.abs(weights) > _THRESHOLD)
    lower = math.fsum(bounds)
    lower -= 8 * np.finfo(float).eps * (abs(value) + math.fsum(map(abs, bounds)))
    if value - lower > 1e-5:
        raise RuntimeError(f'Unable to certify block sum: gap={value-lower:g}, nodes={nodes}')
    return dict(weights=weights.tolist(), objective=float(value),
                certified_lower_bound=float(min(lower, value)), nodes_explored=nodes)


def _gram_blocks(X, y, ridge, M, fee):
    """Use small Gram blocks without dropping the cross-block quadratic."""
    p = X.shape[1]
    if p < 4 or M <= _THRESHOLD:
        return None
    H = X.T @ X + ridge * np.eye(p)
    scale = np.sqrt(H.diagonal())
    corr = np.abs(H / scale / scale[:, None])
    remaining = set(range(p))
    groups = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        block = [root]
        for j in block:
            for k in sorted(remaining):
                if corr[j, k] > .015:
                    remaining.remove(k)
                    block.append(k)
            if len(block) > 3:
                return None
        groups.append(np.array(block))
    cross = H.copy()
    for cols in groups:
        cross[np.ix_(cols, cols)] = 0
    # Symmetric diagonal dominance makes K PSD, including the Gram roundoff allowance.
    eps = np.finfo(float).eps
    absolute_gram = np.abs(X).T @ np.abs(X)
    safety = 16 * eps * max(1, len(y)) * (absolute_gram.sum(axis=1) + ridge)
    shift = np.abs(cross).sum(axis=1) + safety
    if np.max(shift / H.diagonal()) > .08:
        return None
    radius = np.abs(y) + M * np.sum(np.abs(X), axis=1)
    pad = 64 * eps * (p + len(y)) * (radius @ radius + p * (fee + ridge * M*M))
    blocks = []
    for cols in groups:
        D = H[np.ix_(cols, cols)] - np.diag(shift[cols])
        # Keep the block solves well-conditioned; only K needs to be PSD for validity.
        if np.linalg.eigvalsh(D)[0] < .08 * np.max(D.diagonal()):
            return None
        k = len(cols)
        modes = (np.arange(1 << k)[:, None] >> np.arange(k)) & 1
        lim = np.where(modes, M, _THRESHOLD)
        faces = np.array(list(itertools.product((-1, 0, 1), repeat=k)))
        origin = lim[:, None, :] * faces[None, :, :]
        jac = np.zeros((len(faces), k, k))
        for j, face in enumerate(faces):
            ix = np.flatnonzero(face == 0)
            if len(ix):
                jac[j][np.ix_(ix, ix)] = np.linalg.inv(D[np.ix_(ix, ix)])
        intercept = origin - origin - np.einsum('fij,mfj->mfi', jac, origin @ D)
        blocks.append((cols, D, modes, lim, jac, intercept))

    upper = .5 * (y @ y)
    best = np.zeros(p)
    def update(z):
        nonlocal upper, best
        r = X @ z - y
        value = .5 * (r @ r + ridge * (z @ z)) + fee * np.count_nonzero(np.abs(z) > _THRESHOLD)
        if value < upper:
            upper, best = value, z.copy()

    def evaluate(state, start):
        z = start.copy()
        strongest = -np.inf
        for iteration in range(10):
            r = X @ z - y
            g = X.T @ r + ridge * z
            # Tangent to the smooth remainder after removing the local quadratics.
            local_quadratic = 0.
            for cols, D, *_ in blocks:
                dz = D @ z[cols]
                g[cols] -= dz
                local_quadratic += z[cols] @ dz
            base = .5 * (r @ r + ridge * (z @ z) - local_quadratic) - g @ z
            local = []
            proposal = np.empty(p)
            for cols, D, modes, lim, jac, intercept in blocks:
                limits = lim[:, None, :]
                w = np.clip(intercept - (jac @ g[cols])[None, :, :], -limits, limits)
                a = w @ D
                q = np.sum((.5*a + g[cols])*w, axis=2) + fee*modes.sum(axis=1)[:, None]
                grad = a + g[cols]
                lowers = np.max(q - np.sum(grad*w + np.abs(grad)*limits, axis=2), axis=1)
                valid = np.all((state[cols] == 0) | (modes == (state[cols] > 0)), axis=1)
                lowers[~valid] = np.inf
                q[~valid] = np.inf
                m, f = np.unravel_index(np.argmin(q), q.shape)
                proposal[cols] = w[m, f]
                local.append((cols, modes, lowers))
            update(proposal)
            bound = base + math.fsum(float(np.min(a)) for _, _, a in local) - pad
            if bound > strongest:
                strongest, details = bound, local
            if strongest >= upper - 8e-6 or np.max(np.abs(proposal-z)) < 1e-12:
                break
            z = proposal if iteration < 2 else .5*(z + proposal)
        return strongest, details, proposal

    queue = []
    closed = np.inf
    nodes = 0
    target = 8e-6
    def enter(state, z, parent):
        nonlocal nodes, closed
        low, details, start = evaluate(state, z)
        # Each block's conditional bound is computed from the same tangent.
        lower = max(parent, low)
        nodes += 1
        if lower >= upper - target or not np.any(state == 0):
            closed = min(closed, lower)
        else:
            heapq.heappush(queue, (lower, nodes, state, start, low, details))

    enter(np.zeros(p, dtype=np.int8), best, -np.inf)
    while queue and nodes + 2 <= 599 and upper - min(closed, queue[0][0]) > target:
        lower, _, state, z, low, details = heapq.heappop(queue)
        if lower >= upper - target:
            closed = min(closed, lower)
            continue
        choices = []
        for cols, modes, b in details:
            bias = b - np.min(b)
            for i, j in enumerate(cols):
                if state[j] == 0:
                    costs = [max(lower, low + np.min(bias[modes[:, i] == on]) - pad) for on in (0, 1)]
                    choices.append((max(costs) >= upper-target, min(costs), max(costs), j, costs))
        _, _, _, j, costs = max(choices)
        for value, bound in zip((-1, 1), costs):
            if bound >= upper - target:
                nodes += 1
                closed = min(closed, bound)
            else:
                child = state.copy()
                child[j] = value
                enter(child, z, bound)
    lower = min(closed, queue[0][0] if queue else np.inf)
    if upper - lower > 1e-5:
        raise RuntimeError(f'Unable to certify Gram blocks: gap={upper-lower:g}, nodes={nodes}')
    return dict(weights=best.tolist(), objective=float(upper),
                certified_lower_bound=float(lower), nodes_explored=nodes)


def _solve_block(X, y, ridge, M, fee, budget):
    p = X.shape[1]
    t = min(M, _THRESHOLD)
    best = np.zeros(p)
    upper = .5 * (y @ y)
    if not p:
        return dict(weights=[], objective=float(upper), certified_lower_bound=float(upper - 8*np.finfo(float).eps*upper*max(1, len(y))), nodes_explored=1)
    H = X.T @ X + ridge * np.eye(p)
    h = X.T @ y
    d = _minorant(H, ridge)
    K = H - np.diag(d)
    step = 1. / max(1e-100, np.linalg.eigvalsh(K)[-1] * 1.000001, np.max(d) * .05)
    # On each positive half-axis the envelope follows a parabola to t,
    # a chord to b, and the charged parabola beyond b.
    b = np.minimum(M, t + np.sqrt(2 * fee / d))
    slope = .5 * d * (b + t) + fee / np.where(b > t, b - t, 1.)
    if M == t:
        slope[:] = np.inf
    # Conservative dot-product and cancellation allowance for node bounds.
    Xe = np.sum(np.abs(X), axis=1) * M
    magnitude = np.sum((np.abs(y) + Xe)**2) + p * (ridge * M**2 + fee)
    pad = 48 * np.finfo(float).eps * max(p + len(y), 1) * magnitude
    tol = 2e-7
    polished = set()

    def update(w):
        nonlocal upper, best
        r = X @ w - y
        value = .5 * (r @ r + ridge * (w @ w)) + fee * np.count_nonzero(np.abs(w) > _THRESHOLD)
        if value < upper:
            upper, best = value, w.copy()

    def polish(selected, z):
        key = tuple(selected)
        if key in polished:
            return
        polished.add(key)
        limits = np.where(selected, M, t)
        w = np.clip(z, -limits, limits)
        use = np.flatnonzero(selected)
        if len(use):
            G = H[np.ix_(use, use)]
            v = h[use]
            a = np.linalg.solve(G, v)
            if np.max(np.abs(a)) <= M:
                w[use] = a
        def fg(a):
            r = X @ a - y
            return .5 * (r @ r + ridge * (a @ a)), X.T @ r + ridge * a
        res = minimize(fg, w, jac=True, method='L-BFGS-B', bounds=list(zip(-limits, limits)),
                       options={'ftol': 1e-15, 'gtol': 2e-10, 'maxiter': 250})
        update(np.clip(res.x, -limits, limits))

    def evaluate(state, start):
        free = state == 0
        inactive = state < 0
        active = state > 0
        limit = np.where(inactive, t, M)

        def bound(z):
            r = X @ z - y
            g = X.T @ r + (ridge - d) * z
            # Minorize the smooth term after removing the diagonal quadratic.
            base = .5 * (r @ r + np.dot(ridge - d, z*z)) - g @ z
            a = np.abs(g)
            u = np.minimum(a / d, t)
            v = np.minimum(a / d, M)
            off = (.5 * d * u - a) * u
            on = (.5 * d * v - a) * v + fee
            terms = np.where(free, np.minimum(off, on), np.where(active, on, off))
            return float(base + np.sum(terms) - pad)

        z = np.clip(start, -limit, limit)
        v = z.copy()
        momentum = 1.
        lower = -np.inf
        for k in range(320):
            q = v - step * (K @ v - h)
            aq = np.abs(q)
            u = np.minimum(aq / (1 + step*d), limit)
            if np.any(free):
                mid = np.clip(aq - step*slope, t, b)
                hull = np.where(u < t, u, np.where(u > b, u, mid))
                u = np.where(free, hull, u)
            nxt = np.copysign(u, q)
            if k % 12 == 0:
                lower = max(lower, bound(nxt))
                if lower >= upper - tol or np.max(np.abs(nxt-z)) < 2e-12:
                    z = nxt
                    break
            nm = .5 * (1 + np.sqrt(1 + 4 * momentum*momentum))
            # Gradient restart avoids oscillation near a kink.
            if (v - nxt) @ (nxt - z) > 0:
                nm = 1.
            v = nxt + ((momentum - 1.) / nm if nm > 1 else 0.) * (nxt-z)
            z, momentum = nxt, nm
        lower = max(lower, bound(z))
        if lower < upper - tol:
            absz = np.abs(z)
            frac = np.clip((absz - t) / np.where(b > t, b-t, 1.), 0., 1.)
            polish(active | (free & (frac >= .5)), z)
            polish(active | (free & (absz > t)), z)
            update(z)
        if not np.any(free) and lower < upper - tol:
            # At a fully fixed node, the smooth subproblem has box KKT bounds.
            def fg(w):
                r = X @ w - y
                return .5 * (r @ r + ridge * (w @ w)), X.T @ r + ridge * w
            result = minimize(fg, z, jac=True, bounds=list(zip(-limit, limit)), method='L-BFGS-B',
                              options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
            z = np.clip(result.x, -limit, limit)
            update(z)
            lower = max(lower, bound(z))
        if not np.any(free):
            return lower, z, -1
        # Envelope discrepancy, with fractional impact as a tie breaker.
        absz = np.abs(z)
        mix = np.clip((absz - t) / np.where(b > t, b-t, 1.), 0., 1.)
        gap = (1-mix) * fee - .5 * d * np.maximum(absz-t, 0.) * np.maximum(b-absz, 0.)
        score = np.where((absz > t) & (absz < b), gap + fee * mix, fee * mix * (1-mix))
        j = int(np.argmax(np.where(free, score, -np.inf)))
        return lower, z, j

    root = np.zeros(p, dtype=np.int8) if M > t else -np.ones(p, dtype=np.int8)
    polish(root == 0, best)
    polish(root > 0, best)
    queue = []
    closed = np.inf
    nodes = 0

    def enter(state, start, parent):
        nonlocal nodes, closed
        lower, z, j = evaluate(state, start)
        lower = max(lower, parent)
        nodes += 1
        if j < 0 or lower >= upper - tol:
            closed = min(closed, lower)
        else:
            heapq.heappush(queue, (lower, nodes, state, z, j))

    enter(root, best, -np.inf)
    while queue and nodes + 2 <= budget and upper - min(closed, queue[0][0]) > tol:
        lower, _, state, z, j = heapq.heappop(queue)
        if lower >= upper - tol:
            closed = min(closed, lower)
            continue
        for value in (-1, 1):
            child = state.copy()
            child[j] = value
            enter(child, z, lower)
    lower = min(closed, queue[0][0] if queue else np.inf)
    if upper - lower > 1e-5:
        raise RuntimeError(f'Unable to certify within node limit: gap={upper-lower:g}, nodes={nodes}')
    return dict(weights=best.tolist(), objective=float(upper),
                certified_lower_bound=float(lower), nodes_explored=nodes)
