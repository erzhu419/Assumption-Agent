"""Box-constrained sparse ridge regression with a support-partition proof."""
import heapq

import numpy as np
from scipy.optimize import minimize


def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l2, l0, M = (float(problem[k]) for k in ('lam2', 'lam0', 'M'))
    G = X.T @ X
    h = X.T @ y
    p = h.size
    scale = np.sqrt(np.maximum(G.diagonal(), np.finfo(float).tiny))
    eig = np.linalg.eigvalsh((G / scale[:, None]) / scale[None, :])[0]
    d = max(0., eig - 1e-10) * G.diagonal()
    H = G.copy()
    H.flat[::p+1] -= d
    q = l2 + d
    Q = G.diagonal() + l2
    t = np.minimum(M, np.sqrt(2*l0/q))
    slope = l0/t + .5*q*t
    tol = 5e-7
    incumbent = np.zeros(p)
    upper = .5 * float(y @ y)
    pad = 1e-10 + 64*np.finfo(float).eps*(upper + p*l0)

    def box_fit(support):
        # Solve the smooth quadratic on a support. An active-set step solves
        # for all free coefficients together, including near-null directions.
        idx = np.flatnonzero(support)
        w = np.zeros(p)
        k = len(idx)
        if not k:
            return w
        B = G[np.ix_(idx, idx)].copy()
        B.flat[::k+1] += l2
        b = h[idx]
        v = np.linalg.solve(B, b)
        if np.max(np.abs(v)) <= M:
            w[idx] = v
            return w
        v = np.clip(v, -M, M)
        bound = np.abs(v) == M
        for _ in range(8*k*k):
            free = ~bound
            u = v.copy()
            if free.any():
                u[free] = np.linalg.solve(B[np.ix_(free, free)],
                                          b[free] - B[np.ix_(free, bound)] @ v[bound])
            hit = free & (np.abs(u) > M)
            if hit.any():
                indices = np.flatnonzero(hit)
                wall = np.copysign(M, u[indices])
                frac = (wall - v[indices]) / (u[indices] - v[indices])
                j = int(np.argmin(frac))
                v = np.clip(v + frac[j]*(u-v), -M, M)
                v[indices[j]] = wall[j]
                bound[indices[j]] = True
                continue
            v = u
            violation = np.where(bound, np.sign(v)*(B @ v-b), -np.inf)
            j = int(np.argmax(violation))
            if violation[j] <= 1e-13 * (1 + np.max(np.abs(b))):
                break
            bound[j] = False
        w[idx] = v
        return w

    def improve(support):
        nonlocal incumbent, upper
        center = box_fit(support)
        w = np.where(np.abs(center) > 1e-8, center, 0.)
        residual = y-X@w
        value = .5*float(residual@residual) + .5*l2*float(w@w) + l0*np.count_nonzero(w)
        if value < upper:
            incumbent, upper = w, value
        return center

    def witness(w, active, free):
        residual = y-X@w
        s = np.abs(X.T@residual + d*w)
        b = np.minimum(M, s/q)
        price = l0 - b*(s-.5*q*b)
        lb = .5*float(residual@residual) + float(residual@(X@w)) + .5*float((d*w)@w)
        lb += float(price[active].sum()+np.minimum(0., price[free]).sum())
        return lb-pad, price

    def refine_envelope(w, zero, active):
        # Split the optional coordinates into positive and negative parts.
        # Convexity and monotonicity make cancellation unnecessary.
        a = np.flatnonzero(active)
        u = np.flatnonzero(~(zero | active))
        na, nu = len(a), len(u)
        idx = np.r_[a, u, u]
        sign = np.r_[np.ones(na+nu), -np.ones(nu)]
        opt = np.arange(na, na+2*nu)
        qi, ti, si = q[idx], t[idx], slope[idx]
        start = np.r_[w[a], np.maximum(0., w[u]), np.maximum(0., -w[u])]
        bounds = [(-M,M)]*na + [(0.,M)]*(2*nu)

        def unpack(v):
            x = np.zeros(p)
            x[a] = v[:na]
            x[u] = v[na:na+nu]-v[na+nu:]
            return x

        def fun(v):
            x = unpack(v)
            hx = H @ x
            linear = v[opt] < ti[opt]
            value = .5*(x @ hx) - h @ x + .5*np.dot(qi[:na],v[:na]**2)
            value += np.sum(np.where(linear, si[opt]*v[opt], l0+.5*qi[opt]*v[opt]**2))
            g = qi*v
            g[opt] = np.where(linear, si[opt], g[opt])
            return value, g + sign*(hx-h)[idx]

        result = minimize(fun, start, jac=True, bounds=bounds, method='L-BFGS-B',
                          options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 400, 'maxcor': min(len(idx), 30)})
        return unpack(result.x)

    def node(zero, active, start):
        free = ~(zero | active)
        if not free.any():
            w = improve(active)
            lb, price = witness(w, active, free)
            return lb, w, price
        w = start.copy()
        w[zero] = 0.
        r = h-H@w
        idx = np.flatnonzero(~zero)
        for k in range(220):
            delta = 0.
            for j in idx:
                z = r[j] + H[j,j]*w[j]
                az = abs(z)
                if active[j] or az >= slope[j] + H[j,j]*t[j]:
                    u = min(M, az/Q[j])
                elif az <= slope[j]:
                    u = 0.
                else:
                    u = min(t[j], (az-slope[j])/H[j,j])
                v = u if z >= 0 else -u
                a = v-w[j]
                if a:
                    w[j] = v
                    r -= a*H[:,j]
                    delta = max(delta, abs(a))
            if delta <= 2e-10*(1+M):
                break
        lb, price = witness(w, active, free)
        if k == 219 and lb < upper-tol:
            trial = refine_envelope(w, zero, active)
            trial_lb, trial_price = witness(trial, active, free)
            if trial_lb > lb:
                lb, price, w = trial_lb, trial_price, trial
        if lb < upper-tol:
            improve(active | (free & (np.abs(w) >= .5*t)))
            if lb < upper-tol:
                support = active | (free & (price < 0))
                if np.any(support != (active | (free & (np.abs(w) >= .5*t)))):
                    improve(support)
        return lb, w, price

    leaves = []
    queue = []
    serial = 0

    def keep(zero, active, lb, w):
        leaves.append({'fixed_zero': zero, 'fixed_active': active, 'lb': float(lb),
                       'minorant_diagonal': d, 'minorant_center': w})

    def push(zero, active, start):
        nonlocal serial
        lb, w, price = node(zero, active, start)
        if lb >= upper-tol:
            keep(zero, active, lb, w)
        else:
            serial += 1
            heapq.heappush(queue, (lb, serial, zero, active, w, price))

    empty = np.zeros(p, dtype=bool)
    push(empty, empty, incumbent)
    while queue:
        lb, _, zero, active, w, price = heapq.heappop(queue)
        if lb >= upper-tol:
            keep(zero, active, lb, w)
            continue
        free = ~(zero | active)
        if not free.any():
            raise RuntimeError('Box quadratic did not converge to certification tolerance')
        x = np.abs(w)
        envelope = np.where(x < t, slope*x, l0+.5*q*x*x)
        # The smaller of the two disjunctive losses promotes balanced splits.
        loss = np.minimum(envelope, l0+.5*q*x*x-envelope)
        j = int(np.argmax(np.where(free, loss, -np.inf)))
        if len(leaves) + len(queue) + 2 > 2048:
            raise RuntimeError('Search exceeded the 2048-leaf certificate budget')
        z = zero.copy()
        z[j] = True
        a = active.copy()
        a[j] = True
        push(z, active, w)
        push(zero, a, w)
    lower = min(upper, min(leaf['lb'] for leaf in leaves))
    return {'weights': incumbent, 'objective': float(upper),
            'certified_lower_bound': float(lower), 'subcube_leaves': leaves}
