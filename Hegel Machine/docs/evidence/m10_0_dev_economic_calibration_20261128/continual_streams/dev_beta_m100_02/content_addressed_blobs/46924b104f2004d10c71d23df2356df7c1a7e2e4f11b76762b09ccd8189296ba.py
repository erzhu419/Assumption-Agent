"""Finite-volume transport inversion and a box-simplex KKT certificate."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin, wall):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = len(kappa)
    i = np.arange(n)
    left = (kappa + np.roll(kappa, 1)) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + np.roll(kappa, -1)) / (2 * h**2) + np.maximum(-velocity, 0) / h
    loss = np.zeros(n)
    if wall != 'periodic':
        left[0] = right[-1] = 0
        right[0], left[-1] = kappa[0] / h**2, kappa[-1] / h**2
        loss[[0, -1]] = robin / h if wall == 'open' else 0
    return csc_matrix((np.r_[left + right + loss, -left, -right],
                       (np.tile(i, 3), np.r_[i, (i-1) % n, (i+1) % n])), shape=(n,n))


def _responses(study, walls=('reflect', 'reflect')):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'], walls[0])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'], walls[1])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + diags(reaction)
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), -1).T
    i, j = np.asarray(study['sensor_ij']).T
    sensors = i + nx * j
    readings = []
    for shift in study['mu_shifts']:
        fields = splu((operator + shift * identity).tocsc()).solve(rhs)
        readings.append(fields[sensors])
    return np.concatenate(readings), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, upper, total):
    n = len(c)
    if total == 0:
        return np.zeros(n), float(np.min(c))
    if total == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w + c))

    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)  # -1 lower bound, +1 upper bound
    tol = 128 * np.finfo(float).eps * max(1., np.max(np.abs(c)), upper * np.linalg.norm(H, np.inf))
    for _ in range(20 * n * n):
        g = H @ w + c
        free = np.flatnonzero(active == 0)
        if free.size:
            # Newton direction on the current face, constrained to sum to zero.
            f = len(free)
            kkt = np.empty((f + 1, f + 1))
            kkt[:f, :f] = H[np.ix_(free, free)]
            kkt[:f, f] = -1
            kkt[f, :f] = 1
            kkt[f, f] = 0
            step = np.linalg.solve(kkt, np.r_[-g[free], 0.])
            delta, nu = step[:-1], step[-1]
            proposed = w[free] + delta
            fraction = np.ones(f)
            low = (proposed < 0) & (delta < 0)
            high = (proposed > upper) & (delta > 0)
            fraction[low] = -w[free[low]] / delta[low]
            fraction[high] = (upper - w[free[high]]) / delta[high]
            hit = int(np.argmin(fraction))
            w[free] += fraction[hit] * delta
            if fraction[hit] < 1:
                j = free[hit]
                active[j] = -1 if low[hit] else 1
                w[j] = 0 if low[hit] else upper
                continue
            g = H @ w + c
        else:
            nu = (np.max(g[active > 0]) + np.min(g[active < 0])) / 2
        violation = active * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    S, terminal_modes = _responses(cal_item)
    y = np.asarray(cal_item['y_obs_raw']).ravel()
    Z = np.asarray(cal_item['Z_nuisance'])
    ridge = cal_item['source_ridge']
    upper, total = cal_item['u_box'], cal_item['total_budget']
    data = np.column_stack((S, y))
    coefficients = np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]),
                                   Z.T @ data)
    residualized = data - Z @ coefficients
    A, b = residualized[:, :-1], residualized[:, -1]
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_simplex(H, -(A.T @ b), upper, total)
    residual = A @ w - b
    g = A.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = g - nu
    dual = objective - g @ w + nu * total + upper * np.minimum(0, reduced).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': float(objective),
                                'dual_lower_bound': float(dual)}}
