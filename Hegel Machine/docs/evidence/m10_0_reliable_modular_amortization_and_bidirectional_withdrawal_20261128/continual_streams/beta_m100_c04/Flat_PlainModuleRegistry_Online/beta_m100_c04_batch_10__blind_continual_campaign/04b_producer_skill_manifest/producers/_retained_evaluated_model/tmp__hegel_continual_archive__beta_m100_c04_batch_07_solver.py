"""Transport and source reconstruction in calibrated chamber coordinates."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, spacing):
    """Row-local upwind fluxes, including the last-to-first periodic face."""
    n = len(kappa)
    face = (kappa + np.roll(kappa, -1)) / (2 * spacing**2)
    left = np.roll(face, 1) + np.maximum(velocity, 0) / spacing
    right = face + np.maximum(-velocity, 0) / spacing
    i = np.arange(n)
    return sparse.csc_matrix(
        (np.concatenate((left + right, -left, -right)),
         (np.tile(i, 3), np.concatenate((i, (i - 1) % n, (i + 1) % n)))),
        shape=(n, n),
    )


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = study['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = study['sensor_ij']
    sensor_rows = ij[:, 0] + nx * ij[:, 1]
    sampled = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


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


def solve_downstream(cal_item: dict) -> dict:
    """Reconstruct sources and terminal field and certify the profiled objective."""
    study = cal_item
    sensors, terminal_modes = _forward(study)
    z = study['Z_nuisance']
    readings = study['y_obs_raw']
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    combined = np.column_stack((sensors, readings))
    profiled = combined - z @ np.linalg.solve(gram, z.T @ combined)
    s, y = profiled[:, :-1], profiled[:, -1]
    cap, budget, ridge = (study[key] for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced).tolist(),
            'lam_upper': np.maximum(0.0, -reduced).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(dual),
        },
    }
