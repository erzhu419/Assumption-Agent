"""Branch-and-bound with a separable, diagonally strengthened quadratic minorant."""

import heapq

import numpy as np


def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l0, l2, M = (float(problem[k]) for k in ('lam0', 'lam2', 'M'))
    p = X.shape[1]
    G = X.T @ X
    xy = X.T @ y
    diag = np.diag(G).copy()
    bits = [1 << j for j in range(p)]
    # Exact zero columns do not constrain the curvature of the remaining block.
    nonzero = diag > 0
    d = np.zeros(p)
    if np.any(nonzero):
        norm = np.sqrt(diag[nonzero])
        corr = (G[np.ix_(nonzero, nonzero)] / norm) / norm[:, None]
        eig = np.linalg.eigvalsh(corr)[0]
        d[nonzero] = max(0.0, eig - 1e-12) * diag[nonzero]
    R = G - np.diag(d)
    Rdiag = np.diag(R).copy()
    h = d + l2
    t = np.minimum(M, np.sqrt(2 * l0 / h))
    slope = l0 / t + 0.5 * h * t
    threshold = 1e-8
    target = 8e-6

    def value(w):
        r = y - X @ w
        return float(0.5 * (r @ r + l2 * (w @ w)) + l0 * np.count_nonzero(np.abs(w) > threshold))

    best_w = np.zeros(p)
    best = value(best_w)
    tried = set()

    def incumbent(mask):
        nonlocal best, best_w
        if mask in tried:
            return
        tried.add(mask)
        ix = [j for j, b in enumerate(bits) if mask & b]
        if not ix:
            return
        B = G[np.ix_(ix, ix)] + l2 * np.eye(len(ix))
        rhs = xy[ix]
        u = np.linalg.solve(B, rhs)
        if np.max(np.abs(u)) > M:
            u = np.clip(u, -M, M)
            for _ in range(300):
                delta = 0.0
                for j in range(len(ix)):
                    v = np.clip(u[j] + (rhs[j] - B[j] @ u) / B[j, j], -M, M)
                    delta = max(delta, abs(v - u[j]))
                    u[j] = v
                if delta <= 1e-12 * max(1.0, M):
                    break
        w = np.zeros(p)
        w[ix] = u
        w[np.abs(w) <= threshold] = 0.0
        f = value(w)
        if f < best:
            best, best_w = f, w

    def relax(zero, active, start):
        c = start.copy()
        indices = []
        free = []
        for j, bit in enumerate(bits):
            if zero & bit:
                c[j] = 0.0
            else:
                is_free = not bool(active & bit)
                indices.append((j, is_free))
                if is_free:
                    free.append(j)
        # Minimize the convex envelope via exact scalar coordinate updates.
        # At every iterate, the tangent minorant is already a valid certificate.
        for _ in range(180):
            delta = 0.0
            for j, is_free in indices:
                a = xy[j] - R[j] @ c + Rdiag[j] * c[j]
                q = abs(a)
                if not is_free:
                    u = q / (diag[j] + l2)
                elif q <= slope[j]:
                    u = 0.0
                elif q <= slope[j] + Rdiag[j] * t[j]:
                    u = (q - slope[j]) / Rdiag[j]
                elif t[j] >= M:
                    u = M
                else:
                    u = q / (diag[j] + l2)
                v = min(M, u) if a >= 0.0 else -min(M, u)
                delta = max(delta, abs(v - c[j]))
                c[j] = v
            if delta <= 2e-11 * max(1.0, M):
                break
        xc = X @ c
        r = y - xc
        a = np.abs(X.T @ r + d * c)
        b = np.minimum(M, a / h)
        net = l0 - b * (a - 0.5 * h * b)
        aterm = sum(net[j] for j, bit in enumerate(bits) if active & bit)
        uterm = sum(min(0.0, net[j]) for j in free)
        constant = 0.5 * (r @ r) + r @ xc + 0.5 * ((d * c) @ c)
        lb = float(constant + aterm + uterm)
        return lb, c, free

    leaves = []
    queue = []
    sequence = 0

    def leaf(z, a, lb, c):
        leaves.append({'fixed_zero': [bool(z & b) for b in bits],
                       'fixed_active': [bool(a & b) for b in bits],
                       'lb': lb,
                       'minorant_diagonal': d.copy(),
                       'minorant_center': c})

    def node(z, a, start):
        nonlocal sequence
        lb, c, free = relax(z, a, start)
        if lb < best - target:
            frac = np.minimum(np.abs(c) / t, 1.0)
            rounded = a | sum(bits[j] for j in free if frac[j] >= 0.5)
            incumbent(rounded)
            if not queue and not leaves and z == a == 0:
                for j in range(p):
                    incumbent(rounded ^ bits[j])
            if free and lb < best - target:
                # Near the midpoint both decisions remove part of the envelope gap.
                j = max(free, key=lambda j: min(frac[j], 1.0 - frac[j]))
                sequence += 1
                heapq.heappush(queue, (lb, sequence, z, a, c, j))
                return
        leaf(z, a, lb, c)

    node(0, 0, best_w)
    while queue:
        lb, _, z, a, c, j = heapq.heappop(queue)
        if lb >= best - target:
            leaf(z, a, lb, c)
        else:
            bit = bits[j]
            node(z | bit, a, c)
            node(z, a | bit, c)
    lower = min(best, min(item['lb'] for item in leaves))
    return {'weights': best_w, 'objective': best,
            'certified_lower_bound': lower, 'subcube_leaves': leaves}
