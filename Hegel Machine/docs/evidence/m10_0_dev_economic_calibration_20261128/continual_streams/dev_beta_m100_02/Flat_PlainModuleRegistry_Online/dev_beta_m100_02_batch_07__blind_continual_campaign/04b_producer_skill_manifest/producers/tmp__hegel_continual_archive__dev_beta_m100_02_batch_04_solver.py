"""Transport inversion for a periodic x axis and reflecting y walls."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    if not periodic:
        upper[0] = -kappa[0] / h**2
        lower[-1] = -kappa[-1] / h**2
    diagonal = np.r_[-upper[0], -lower[:-1] - upper[1:], -lower[-1]]
    if not periodic:
        return diags([lower, diagonal, upper], [-1, 0, 1], format='csc')
    wrap = (kappa[0] + kappa[-1]) / (2 * h**2)
    left = wrap + max(velocity[0], 0) / h
    right = wrap + max(-velocity[-1], 0) / h
    diagonal[0] += left
    diagonal[-1] += right
    n = len(kappa)
    return diags([lower, diagonal, upper, -left, -right],
                 [-1, 0, 1, n-1, 1-n], format='csc')


def _responses(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), True)
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), False)
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)
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
    """Feasible active set for a strictly convex box-simplex quadratic."""
    n = len(c)
    if total == 0:
        return np.zeros(n), float(np.min(c))
    if total == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w + c))
    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(np.linalg.norm(H, np.inf) * upper, np.linalg.norm(c, np.inf), 1.)
    for _ in range(max(100, 20 * n * n)):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if len(free):
            rhs = -c[free] - H[np.ix_(free, fixed)] @ w[fixed]
            reduced = H[np.ix_(free, free)]
            x, v = np.linalg.solve(reduced, np.column_stack((rhs, np.ones(len(free))))).T
            nu = (total - w[fixed].sum() - x.sum()) / v.sum()
            candidate = x + nu * v
            delta = candidate - w[free]
            crosses = np.where((candidate < 0) & (delta < 0), -1,
                               np.where((candidate > upper) & (delta > 0), 1, 0))
            bad = np.flatnonzero(crosses)
            if len(bad):
                ratios = (np.where(crosses[bad] > 0, upper, 0.) - w[free[bad]]) / delta[bad]
                pos = np.argmin(ratios)
                j = bad[pos]
                w[free] = np.clip(w[free] + ratios[pos] * delta, 0., upper)
                active[free[j]] = crosses[j]
                w[free[j]] = upper if crosses[j] > 0 else 0.
                continue
            w[free] = candidate
        g = H @ w + c
        if not len(free):
            lo = np.max(g[active > 0], initial=-np.inf)
            hi = np.min(g[active < 0], initial=np.inf)
            nu = (lo + hi) / 2
        violation = active * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    S, U_last = _responses(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    y = np.asarray(cal_item['y_obs_raw']).ravel()
    ridge = cal_item['source_ridge']
    upper, total = cal_item['u_box'], cal_item['total_budget']
    data = np.column_stack((S, y))
    nuisance = np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]),
                               Z.T @ data)
    residualized = data - Z @ nuisance
    A, b = residualized[:, :-1], residualized[:, -1]
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_simplex(H, -(A.T @ b), upper, total)
    residual = A @ w - b
    gradient = A.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = gradient - nu
    dual = objective - gradient @ w + nu * total + upper * np.minimum(0, reduced).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': float(objective),
                                'dual_lower_bound': float(dual)}}
