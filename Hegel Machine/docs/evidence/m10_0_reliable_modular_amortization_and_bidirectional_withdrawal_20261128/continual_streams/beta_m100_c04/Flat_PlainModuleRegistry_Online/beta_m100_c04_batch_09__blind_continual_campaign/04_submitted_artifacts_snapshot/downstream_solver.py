"""Calibrated transport reconstruction with a box-simplex certificate."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin, wall):
    n = len(kappa)
    if wall == 'periodic':
        face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
        left = np.roll(face, 1) + np.maximum(velocity, 0) / h
        right = face + np.maximum(-velocity, 0) / h
        i = np.arange(n)
        return sparse.csc_matrix((np.r_[left + right, -left, -right],
                                  (np.tile(i, 3), np.r_[i, (i-1)%n, (i+1)%n])), shape=(n, n))
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(velocity[1:-1], 0) / h
    upper[1:] -= np.maximum(-velocity[1:-1], 0) / h
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    leak = robin / h if wall == 'robin' else 0.0
    diagonal = np.r_[leak - upper[0], -lower[:-1] - upper[1:], leak - lower[-1]]
    return sparse.diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _forward(study, walls):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx']/(nx-1), study['robin_x'], walls[0])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly']/(ny-1), study['robin_y'], walls[1])
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = study['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = study['sensor_ij']
    rows = ij[:, 0] + nx * ij[:, 1]
    readings = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(rhs)
        readings.append(field[rows])
    return np.vstack(readings), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)

def _box_simplex(hessian, linear, cap, budget):
    """Feasible active-set QP with the equality multiplier signed as g - nu."""
    n = len(linear)
    if budget == 0:
        return np.zeros(n), float(np.min(linear))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(hessian @ w + linear))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1: lower; +1: upper
    tol = 1e-12 * max(1, np.linalg.norm(linear, np.inf), cap * np.linalg.norm(hessian, np.inf))
    for _ in range(100 * n):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if free.size == 0:
            gradient = hessian @ w + linear
            low = np.where(active == -1, gradient, np.inf).argmin()
            high = np.where(active == 1, gradient, -np.inf).argmax()
            if gradient[high] <= gradient[low] + tol:
                return w, float((gradient[high] + gradient[low]) / 2)
            active[[low, high]] = 0
            continue
        block = hessian[np.ix_(free, free)]
        rhs = -linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed]
        ones = np.ones(free.size)
        base, response = np.linalg.solve(block, np.column_stack((rhs, ones))).T
        nu = (budget - w[fixed].sum() - base.sum()) / response.sum()
        target = base + nu * response
        direction = target - w[free]
        fractions = np.full(free.size, np.inf)
        below = (target < -1e-14) & (direction < 0)
        above = (target > cap + 1e-14) & (direction > 0)
        fractions[below] = -w[free[below]] / direction[below]
        fractions[above] = (cap - w[free[above]]) / direction[above]
        hit = fractions.argmin()
        if fractions[hit] < 1.0:
            w[free] += max(0.0, fractions[hit]) * direction
            active[free[hit]] = 1 if above[hit] else -1
            w[free[hit]] = cap if above[hit] else 0.0
            continue
        w[free] = np.clip(target, 0.0, cap)
        violation = active * (hessian @ w + linear - nu)
        release = violation.argmax()
        if violation[release] <= tol:
            return w, float(nu)
        active[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _fit(study, sensors, readings, z):
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    combined = np.column_stack((sensors, readings))
    nuisance = np.linalg.solve(gram, z.T @ combined)
    profiled = combined - z @ nuisance
    s, y = profiled[:, :-1], profiled[:, -1]
    cap, budget, ridge = (study[key] for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    return w, nu, s, y, nuisance[:, -1] - nuisance[:, :-1] @ w


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    sensors, terminal_modes = _forward(study, ('periodic', 'sealed'))
    cap, budget, ridge = (study[key] for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu, s, y, _ = _fit(study, sensors, study['y_obs_raw'], study['Z_nuisance'])
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced).tolist(),
            'lam_upper': np.maximum(0.0, -reduced).tolist(),
            'primal_objective': objective, 'dual_lower_bound': float(dual),
        },
    }
