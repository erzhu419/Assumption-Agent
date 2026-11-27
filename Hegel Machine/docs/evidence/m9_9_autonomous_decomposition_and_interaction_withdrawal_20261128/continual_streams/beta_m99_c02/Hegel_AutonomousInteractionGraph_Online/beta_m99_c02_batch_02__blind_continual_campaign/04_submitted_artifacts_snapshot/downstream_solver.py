"""Periodic transport-reaction inverse solve in calibrated coordinates."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    # Every row, including each seam endpoint, uses the same conservative
    # face diffusion and row-local upwind rule with cyclic neighbor indices.
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    faces = (kappa + kappa[right]) / (2 * h**2)
    lm = faces[left] + np.maximum(velocity, 0) / h
    rp = faces + np.maximum(-velocity, 0) / h
    return csc_matrix((np.r_[lm + rp, -lm, -rp],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n))


def _responses(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = (_axis(study['kappa_' + a], study['vel_' + a], study['L' + a] / (n - 1))
              for a, n in (('x', nx), ('y', ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(study['V_reaction_2d'].ravel(order='F'))).tocsc()
    ident = eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ij = study['sensor_ij']
    indices = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(rhs)
        blocks.append(fields[indices])
    terminal = np.stack([column.reshape(nx, ny, order='F') for column in fields.T])
    return np.concatenate(blocks, axis=0), terminal


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
    s, terminal = _responses(cal_item)
    z = np.asarray(cal_item['Z_nuisance'], dtype=float)
    y = np.asarray(cal_item['y_obs_raw'], dtype=float)
    observations = np.column_stack((s, y))
    projected = observations - z @ np.linalg.solve(
        z.T @ z + cal_item['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ observations)
    x, r_y = projected[:, :-1], projected[:, -1]
    ridge = cal_item['source_ridge']
    cap, total = cal_item['u_box'], cal_item['total_budget']
    w, nu = _box_simplex(x.T @ x + ridge * np.eye(x.shape[1]), -x.T @ r_y, cap, total)
    residual = x @ w - r_y
    g = x.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    lower = np.maximum(reduced, 0)
    upper = np.maximum(-reduced, 0)
    # Equivalent to J - g @ w + nu * total - cap * sum(upper),
    # with reduced-gradient dot products avoiding cancellation.
    gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - total)
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': objective,
                                'dual_lower_bound': float(objective - gap)}}
