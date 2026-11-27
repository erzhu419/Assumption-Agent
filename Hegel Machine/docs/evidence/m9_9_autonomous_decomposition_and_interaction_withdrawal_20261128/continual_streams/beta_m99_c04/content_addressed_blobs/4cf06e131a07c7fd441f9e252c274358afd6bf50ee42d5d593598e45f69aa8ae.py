"""Transport and source inversion for the periodic-x, sealed-y chamber."""

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    a = (kappa + kappa[left]) / (2 * h**2) + np.maximum(velocity, 0) / h
    b = (kappa + kappa[right]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    if not periodic:
        a[0], b[0] = 0, kappa[0] / h**2
        a[-1], b[-1] = kappa[-1] / h**2, 0
    return coo_matrix((np.r_[a + b, -a, -b],
                       (np.tile(i, 3), np.r_[i, left, right])),
                      shape=(n, n)).tocsc()


def _design(s):
    nx, ny = s['nx'], s['ny']
    ax = _axis(s['kappa_x'], s['vel_x'], s['Lx'] / (nx - 1), periodic=True)
    ay = _axis(s['kappa_y'], s['vel_y'], s['Ly'] / (ny - 1), periodic=False)
    reaction = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    rhs = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
    ix, iy = s['sensor_ij'].T
    blocks = []
    for mu in s['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        blocks.append(fields[ix + nx * iy])
    terminal = fields.T.reshape((-1, ny, nx)).transpose(0, 2, 1)
    return np.vstack(blocks), terminal


def _box_simplex(H, c, cap, budget):
    """Minimize 0.5*w.T@H@w + c@w on the capped simplex."""
    n = len(c)
    w = np.full(n, budget / n)
    if budget == 0:
        return w, float(c.min())
    if budget == n * cap:
        return w, float((H @ w + c).max())
    # -1 is a lower bound, +1 an upper bound, 0 a free variable.
    status = np.zeros(n, dtype=int)
    tol = 128 * np.finfo(float).eps * max(1., np.max(np.abs(c)), cap * np.linalg.norm(H, np.inf))
    for _ in range(100 * n):
        g = H @ w + c
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            lower, upper = np.flatnonzero(status == -1), np.flatnonzero(status == 1)
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            status[[lo, hi]] = 0
            continue
        # Profile out the equality multiplier using the free Hessian.
        x, z = np.linalg.solve(H[np.ix_(free, free)],
                               np.column_stack((g[free], np.ones(free.size)))).T
        nu = (x.sum() + budget - w.sum()) / z.sum()
        step = nu * z - x
        alpha, blocker, side = 1., -1, 0
        for j, d in zip(free, step):
            if d < 0 and w[j] + d < 0:
                ratio, bound = -w[j] / d, -1
            elif d > 0 and w[j] + d > cap:
                ratio, bound = (cap - w[j]) / d, 1
            else:
                continue
            if ratio < alpha:
                alpha, blocker, side = max(0., ratio), j, bound
        w[free] += alpha * step
        if blocker >= 0:
            w[blocker] = 0. if side < 0 else cap
            status[blocker] = side
            continue
        g = H @ w + c
        violation = status * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol:
            return w, float(nu)
        status[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    S, terminal = _design(s)
    Z, y = s['Z_nuisance'], s['y_obs_raw']
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (Pz @ S)
    y = y - Z @ (Pz @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S.T @ y), cap, budget)
    residual = S @ w - y
    J = 0.5 * (residual @ residual + ridge * (w @ w))
    g = S.T @ residual + ridge * w
    d = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(d, 0.).tolist(),
            'lam_upper': np.maximum(-d, 0.).tolist(),
            'primal_objective': float(J),
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(d, 0.).sum()),
        },
    }
