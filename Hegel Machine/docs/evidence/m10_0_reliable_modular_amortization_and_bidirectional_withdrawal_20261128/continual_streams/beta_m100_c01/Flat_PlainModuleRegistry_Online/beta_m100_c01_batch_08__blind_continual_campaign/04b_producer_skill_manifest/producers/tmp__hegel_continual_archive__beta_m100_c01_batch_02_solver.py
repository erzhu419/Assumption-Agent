"""Periodic upwind transport and a box-simplex source inverse problem."""

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    """Apply the row-local stencil to every cell, including the periodic seam."""
    kappa = np.asarray(kappa, dtype=float)
    velocity = np.asarray(velocity, dtype=float) / h
    i = np.arange(kappa.size)
    previous = (i - 1) % i.size
    following = (i + 1) % i.size
    left = (kappa + kappa[previous]) / (2 * h**2) + np.maximum(velocity, 0.0)
    right = (kappa + kappa[following]) / (2 * h**2) + np.maximum(-velocity, 0.0)
    return csc_matrix((np.r_[left + right, -left, -right],
                       (np.tile(i, 3), np.r_[i, previous, following])),
                      shape=(i.size, i.size))


def _transport(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + axis], study['vel_' + axis],
                    study['L' + axis] / (n - 1))
              for axis, n in (('x', nx), ('y', ny))]
    reaction = diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * identity).solve(rhs)
        blocks.append(field[slots])
    return np.vstack(blocks), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, budget):
    """Feasible active-set QP; nu uses the sign convention g - nu."""
    n = len(c)
    if budget == 0 or budget == n * cap:
        w = np.full(n, 0.0 if budget == 0 else cap)
        g = H @ w + c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    tol = 1e-12 * max(1.0, np.max(np.abs(H)) * cap, np.max(np.abs(c)))
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        g = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            state[[i, j]] = 0
            continue
        F = H[np.ix_(free, free)]
        # Solve for the face minimizer, enforcing the remaining sum exactly.
        rhs = -c[free] - H[free] @ np.where(state == 0, 0.0, w)
        ones = np.ones(len(free))
        a, b = np.linalg.solve(F, np.column_stack((rhs, ones))).T
        nu = (budget - np.sum(w[state != 0]) - a.sum()) / b.sum()
        wf = a + nu * b
        step = wf - w[free]
        hits = np.full(len(free), np.inf)
        lower, upper = wf < -tol, wf > cap + tol
        hits[lower] = -w[free][lower] / step[lower]
        hits[upper] = (cap - w[free][upper]) / step[upper]
        k = int(np.argmin(hits))
        if hits[k] <= 1.0:
            w[free] += np.clip(hits[k], 0.0, 1.0) * step
            state[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0.0 if lower[k] else cap
            continue
        w[free] = np.clip(wf, 0.0, cap)
        g = H @ w + c
        violation = state * (g - nu)
        i = int(np.argmax(violation))
        if violation[i] <= tol:
            return w, float(nu)
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Recover source weights from upstream-calibrated observations."""
    S, terminal_modes = _transport(cal_item)
    y = np.asarray(cal_item['y_obs_raw'], dtype=float)
    Z = np.asarray(cal_item['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = cal_item['source_ridge']
    cap, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
