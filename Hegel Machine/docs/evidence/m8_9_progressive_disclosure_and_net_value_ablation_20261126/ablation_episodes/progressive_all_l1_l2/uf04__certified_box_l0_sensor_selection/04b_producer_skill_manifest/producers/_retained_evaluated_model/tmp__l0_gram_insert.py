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
    K = cross + np.diag(shift)
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
        intercept = origin - np.einsum('ijk,mjk->mji', jac.transpose(1, 0, 2), origin @ D) if False else origin - np.einsum('fij,mfj->mfi', jac, origin @ D)
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
        for _ in range(10):
            r = X @ z - y
            g = X.T @ r + ridge * z
            # Tangent to the smooth cross-block remainder.
            for cols, D, *_ in blocks:
                g[cols] -= D @ z[cols]
            base = .5 * (r @ r + ridge * (z @ z) - z @ (K @ z)) - g @ z
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
            z = proposal if _ < 2 else .5*(z + proposal)
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


