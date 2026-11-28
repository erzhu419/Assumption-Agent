"""Periodic transport on both axes, with bounded-simplex source inversion."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    # Every row uses the interior stencil, with neighbors taken modulo n.
    right = (kappa + np.roll(kappa, -1)) / (2 * h**2)
    left = np.roll(right, 1)
    backward = left + np.maximum(velocity, 0) / h
    forward = right + np.maximum(-velocity, 0) / h
    A = diags((-backward[1:], backward + forward, -forward[:-1]),
              (-1, 0, 1), format='lil')
    A[0, -1] -= backward[0]
    A[-1, 0] -= forward[-1]
    return A.tocsc()


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    ij = study['sensor_ij']
    rows = ij[:, 0] + nx * ij[:, 1]
    samples = []
    for mu in study['mu_shifts']:
        field = splu(operator + mu * identity).solve(rhs)
        samples.append(field[rows, :])
    return np.concatenate(samples, axis=0), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


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
    y = np.asarray(cal_item['y_obs_raw'], dtype=float)
    Z = np.asarray(cal_item['Z_nuisance'], dtype=float)
    data = np.column_stack((S, y))
    data -= Z @ np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    A, b = data[:, :-1], data[:, -1]
    ridge = float(cal_item['source_ridge'])
    upper, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_simplex(H, -(A.T @ b), upper, budget)
    residual = A @ w - b
    g = A.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = np.maximum(0., g - nu)
    higher = np.maximum(0., nu - g)
    dual = float(obj - g @ w + nu * budget - upper * higher.sum())
    return {'weights': w,
            'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': lower.tolist(), 'lam_upper': higher.tolist(),
                                'primal_objective': obj, 'dual_lower_bound': dual}}
