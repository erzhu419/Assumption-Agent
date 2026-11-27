"""Solve the calibrated chamber with periodic transport on both axes."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    prev, nxt = (i - 1) % n, (i + 1) % n
    left = (kappa + kappa[prev]) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + kappa[nxt]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, prev, nxt])), shape=(n, n))


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = (_axis(study['kappa_' + axis], study['vel_' + axis], study['L' + axis])
              for axis in 'xy')
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = study['sensor_ij'].T
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[i + nx*j])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, budget):
    """Feasible active-set QP with one equality constraint."""
    n = c.size
    if budget == 0 or cap == 0:
        w = np.zeros(n)
        return w, float(c.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((H @ w + c).max())
    w = np.full(n, budget / n)
    # -1: fixed at zero, 1: fixed at cap, 0: free.
    active = np.zeros(n, dtype=int)
    scale = max(np.linalg.norm(H, ord=np.inf), np.linalg.norm(c, ord=np.inf), 1e-30)
    H, c = H / scale, c / scale
    tol = 64 * np.finfo(float).eps
    for _ in range(100 + 20 * n):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active)
        if free.size:
            m = free.size
            system = np.ones((m + 1, m + 1))
            system[:m, :m] = H[np.ix_(free, free)]
            system[:m, m] = -1
            system[m, m] = 0
            rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
            step = np.linalg.solve(system, rhs)
            target, nu = step[:-1], step[-1]
            hit_low, hit_high = target < -tol * cap, target > cap * (1 + tol)
            if np.any(hit_low | hit_high):
                time = np.full(m, np.inf)
                delta = target - w[free]
                time[hit_low] = -w[free][hit_low] / delta[hit_low]
                time[hit_high] = (cap - w[free][hit_high]) / delta[hit_high]
                hit = time.argmin()
                w[free] = np.clip(w[free] + time[hit] * delta, 0, cap)
                active[free[hit]] = -1 if hit_low[hit] else 1
                w[free[hit]] = 0 if hit_low[hit] else cap
                continue
            w[free] = np.clip(target, 0, cap)
            violation = active * (H @ w + c - nu)
            idx = violation.argmax()
            if violation[idx] <= tol:
                return w, float(nu * scale)
            active[idx] = 0
        else:
            # At a vertex, a transfer can improve the objective only if a
            # capped source has larger gradient than a zero-weight source.
            g = H @ w + c
            lo = np.flatnonzero(active == -1)
            hi = np.flatnonzero(active == 1)
            i, j = lo[g[lo].argmin()], hi[g[hi].argmax()]
            if g[j] <= g[i] + tol:
                return w, float(0.5 * (g[i] + g[j]) * scale)
            active[[i, j]] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    design, terminal_modes = _forward(s)
    z = s['Z_nuisance']
    observations = np.column_stack((design, s['y_obs_raw']))
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    residualized = observations - z @ np.linalg.solve(gram, z.T @ observations)
    a, y = residualized[:, :-1], residualized[:, -1]
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    hessian = a.T @ a + ridge * np.eye(a.shape[1])
    w, nu = _box_simplex(hessian, -a.T @ y, cap, budget)
    residual = a @ w - y
    g = a.T @ residual + ridge * w
    obj = float((residual @ residual + ridge * (w @ w)) / 2)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': float(nu),
            'lam_lower': np.maximum(0, g - nu).tolist(),
            'lam_upper': np.maximum(0, nu - g).tolist(),
            'primal_objective': obj,
            'dual_lower_bound': float(obj - g @ w + nu * budget + cap * np.minimum(0, g - nu).sum()),
        },
    }
