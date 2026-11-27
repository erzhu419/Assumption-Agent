"""Calibrated transport inversion: periodic x seam and reflecting y walls."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axes(s):
    k, v = s['kappa_x'], s['vel_x']
    nx, ny = s['nx'], s['ny']
    dx, dy = s['Lx'] / (nx - 1), s['Ly'] / (ny - 1)
    i = np.arange(nx)
    left, right = (i - 1) % nx, (i + 1) % nx
    faces = (k + k[right]) / (2 * dx**2)
    lower = faces[left] + np.maximum(v, 0) / dx
    upper = faces + np.maximum(-v, 0) / dx
    ax = csc_matrix((np.r_[lower + upper, -lower, -upper],
                     (np.tile(i, 3), np.r_[i, left, right])), shape=(nx, nx))

    k, v = s['kappa_y'], s['vel_y'][1:-1]
    faces = (k[:-1] + k[1:]) / (2 * dy**2)
    lower = faces[:-1] + np.maximum(v, 0) / dy
    upper = faces[1:] + np.maximum(-v, 0) / dy
    first, last = k[0] / dy**2, k[-1] / dy**2
    ay = diags((np.r_[-lower, -last], np.r_[first, lower + upper, last],
                np.r_[-first, -upper]), (-1, 0, 1), shape=(ny, ny), format='csc')
    return ax, ay


def _responses(s):
    nx, ny = s['nx'], s['ny']
    ax, ay = _axes(s)
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ident = eye(nx * ny, format='csc')
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    i, j = s['sensor_ij'].T
    blocks = []
    for shift in s['mu_shifts']:
        fields = splu(operator + shift * ident).solve(rhs)
        blocks.append(fields[i + nx * j])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(hess, linear, cap, total):
    """Primal active set: solve on the current face, then test bound multipliers."""
    n = len(linear)
    if total == 0 or total == n * cap:
        w = np.full(n, 0.0 if total == 0 else cap)
        g = hess @ w + linear
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    bounds = np.zeros(n, dtype=int)
    tol = 128 * np.finfo(float).eps * max(np.linalg.norm(hess, np.inf) * cap, np.linalg.norm(linear, np.inf), 1e-12)
    for _ in range(20 * n + 100):
        free = np.flatnonzero(bounds == 0)
        g = hess @ w + linear
        if len(free) == 0:
            lo, hi = np.flatnonzero(bounds < 0), np.flatnonzero(bounds > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[i] + g[j]) / 2)
            bounds[[i, j]] = 0
            continue
        sub = hess[np.ix_(free, free)]
        rhs = -(linear[free] + hess[free] @ np.where(bounds != 0, w, 0))
        base, response = np.linalg.solve(sub, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (total - w[bounds != 0].sum() - base.sum()) / response.sum()
        target = base + response * nu
        direction = target - w[free]
        fraction = np.full(len(free), np.inf)
        lo, hi = (target < 0) & (direction < 0), (target > cap) & (direction > 0)
        fraction[lo] = -w[free[lo]] / direction[lo]
        fraction[hi] = (cap - w[free[hi]]) / direction[hi]
        hit = np.argmin(fraction)
        if fraction[hit] < 1:
            w[free] += fraction[hit] * direction
            i = free[hit]
            bounds[i] = -1 if lo[hit] else 1
            w[i] = 0.0 if lo[hit] else cap
            continue
        w[free] = np.clip(target, 0, cap)
        g = hess @ w + linear
        violation = bounds * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        bounds[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    S, terminal = _responses(s)
    Z = s['Z_nuisance']
    y = s['y_obs_raw']
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    X = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = s['source_ridge']
    cap, total = s['u_box'], s['total_budget']
    H = X.T @ X + ridge * np.eye(X.shape[1])
    w, nu = _box_simplex(H, -X.T @ r, cap, total)
    e = X @ w - r
    g = X.T @ e + ridge * w
    lo, hi = np.maximum(g - nu, 0), np.maximum(nu - g, 0)
    objective = float(0.5 * (e @ e + ridge * (w @ w)))
    # Complementarity form of J - g@w + nu*total - cap*sum(hi).
    gap = lo @ w + hi @ (cap - w) + nu * (w.sum() - total)
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=1),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': lo.tolist(), 'lam_upper': hi.tolist(),
                                'primal_objective': objective,
                                'dual_lower_bound': float(objective - gap)}}
