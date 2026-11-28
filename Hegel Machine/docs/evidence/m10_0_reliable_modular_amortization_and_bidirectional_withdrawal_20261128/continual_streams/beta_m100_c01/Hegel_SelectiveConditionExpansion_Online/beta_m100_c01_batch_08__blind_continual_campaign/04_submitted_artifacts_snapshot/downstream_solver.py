"""Periodic-x, reflecting-y transport and a certified box-simplex fit."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    face = 0.5 * (kappa + np.roll(kappa, -1)) / h**2
    lower = np.roll(face, 1) + np.maximum(velocity, 0.0) / h
    upper = face + np.maximum(-velocity, 0.0) / h
    op = diags((-lower[1:], lower + upper, -upper[:-1]), (-1, 0, 1), format='lil')
    if periodic:
        op[0, -1] -= lower[0]
        op[-1, 0] -= upper[-1]
    else:
        # Reflecting boundary rows have zero net flux and no wall advection.
        left_wall = kappa[0] / h**2
        right_wall = kappa[-1] / h**2
        op[0, 0], op[0, 1] = left_wall, -left_wall
        op[-1, -1], op[-1, -2] = right_wall, -right_wall
    return op.tocsc()


def _transport(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (n - 1), periodic=(a == 'x'))
              for a, n in (('x', nx), ('y', ny))]
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
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
    """Reconstruct sources from the canonical observations supplied upstream."""
    study = cal_item
    S, U_last = _transport(study)
    y = np.asarray(study['y_obs_raw'], dtype=float)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = study['source_ridge']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
