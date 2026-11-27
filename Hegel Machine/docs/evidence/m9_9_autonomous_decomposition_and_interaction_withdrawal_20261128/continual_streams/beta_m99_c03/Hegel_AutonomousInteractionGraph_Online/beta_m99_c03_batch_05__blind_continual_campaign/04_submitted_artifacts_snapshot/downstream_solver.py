"""Transport with periodic x and reflecting y, followed by source inversion."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_operator(kappa, velocity, length, periodic):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n) if periodic else np.arange(1, n - 1)
    left, right = (i - 1) % n, (i + 1) % n
    minus = (kappa[i] + kappa[left]) / (2 * h**2) + np.maximum(velocity[i], 0) / h
    plus = (kappa[i] + kappa[right]) / (2 * h**2) + np.maximum(-velocity[i], 0) / h
    a = np.zeros((n, n))
    a[i, i], a[i, left], a[i, right] = minus + plus, -minus, -plus
    if not periodic:
        a[0, 0] = kappa[0] / h**2
        a[-1, -1] = kappa[-1] / h**2
        a[0, 1], a[-1, -2] = -a[0, 0], -a[-1, -1]
    return csc_matrix(a)


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis_operator(study['kappa_x'], study['vel_x'], study['Lx'], True)
    ay = _axis_operator(study['kappa_y'], study['vel_y'], study['Ly'], False)
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = study['sensor_ij'].T
    samples = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        samples.append(fields[i + nx * j])
    return np.vstack(samples), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(hessian, linear, cap, budget):
    n = len(linear)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((hessian @ w + linear).max())

    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1 lower, +1 upper, 0 free
    for _ in range(100 * n):
        g = hessian @ w + linear
        free = np.flatnonzero(active == 0)
        if not len(free):
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue

        # Minimize on the current face and stop at the first blocking bound.
        k = len(free)
        system = np.ones((k + 1, k + 1))
        system[:k, :k] = hessian[np.ix_(free, free)]
        system[-1, -1] = 0
        step_and_multiplier = np.linalg.solve(system, np.r_[-g[free], 0.0])
        step = step_and_multiplier[:-1]
        target = w[free] + step
        limits = np.ones(k)
        lower, upper = target < -1e-14, target > cap + 1e-14
        limits[lower] = -w[free[lower]] / step[lower]
        limits[upper] = (cap - w[free[upper]]) / step[upper]
        block = np.argmin(limits)
        alpha = limits[block]
        w[free] += alpha * step
        if alpha < 1:
            i = free[block]
            active[i] = -1 if lower[block] else 1
            w[i] = 0.0 if active[i] < 0 else cap
            continue

        nu = float(-step_and_multiplier[-1])
        reduced = hessian @ w + linear - nu
        violation = active * reduced
        i = np.argmax(violation)
        if violation[i] <= 1e-12:
            return np.clip(w, 0, cap), nu
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    response, last = _forward(study)
    z = study['Z_nuisance']
    y = study['y_obs_raw']  # Canonical coordinates from the upstream transducer.
    ridge, cap, budget = (study[key] for key in ('source_ridge', 'u_box', 'total_budget'))
    projection = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s = response - z @ (projection @ response)
    r = y - z @ (projection @ y)
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ r, cap, budget)
    residual = s @ w - r
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    reduced = g - nu
    dual = float(objective - g @ w + nu * budget + cap * np.minimum(0, reduced).sum())
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0, reduced).tolist(),
            'lam_upper': np.maximum(0, -reduced).tolist(),
            'primal_objective': objective, 'dual_lower_bound': dual,
        },
    }
