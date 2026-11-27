"""Periodic transport-reaction inversion in calibrated sensor coordinates."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    # Each endpoint is a regular periodic row, sharing a face with the other endpoint.
    face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
    west = np.roll(face, 1) + np.maximum(vel, 0.) / h
    east = face + np.maximum(-vel, 0.) / h
    mat = diags((-west[1:], west + east, -east[:-1]), (-1, 0, 1), format='lil')
    mat[0, -1] -= west[0]
    mat[-1, 0] -= east[-1]
    return mat.tocsc()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    axes = [_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                  study['L' + d] / (n - 1))
            for d, n in zip('xy', (nx, ny))]
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx)) + diags(reaction)).tocsc()
    ident = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = np.asfortranarray(modes.transpose(0, 2, 1).reshape(len(modes), -1).T)
    ix, iy = np.asarray(study['sensor_ij']).T
    rows = ix + nx * iy
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * ident).solve(rhs)
        blocks.append(field[rows, :])
    terminal_modes = field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)
    return np.vstack(blocks), terminal_modes


def _box_simplex(hessian, linear, cap, budget):
    """Primal active set, using exact equality-constrained solves on each face."""
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1: lower; +1: upper
    tol = 1e-12 * max(1., np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 20):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status)
        if not len(free):
            g = hessian @ w + linear
            lower, upper = np.flatnonzero(status == -1), np.flatnonzero(status == 1)
            a, b = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[a] >= g[b] - tol:
                return w, float((g[a] + g[b]) / 2)
            status[[a, b]] = 0
            continue
        hff = hessian[np.ix_(free, free)]
        cf = linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed]
        q, t = np.linalg.solve(hff, np.column_stack((-cf, np.ones(len(free))))).T
        nu = (budget - w[fixed].sum() - q.sum()) / t.sum()
        target = q + nu * t
        step = target - w[free]
        ratio = np.full(len(free), np.inf)
        lower, upper = (target < 0) & (step < 0), (target > cap) & (step > 0)
        ratio[lower] = -w[free[lower]] / step[lower]
        ratio[upper] = (cap - w[free[upper]]) / step[upper]
        k = int(ratio.argmin())
        if ratio[k] < 1:
            w[free] += ratio[k] * step
            status[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0. if lower[k] else cap
            continue
        w[free] = target
        g = hessian @ w + linear
        release = status * (g - nu)
        k = int(release.argmax())
        if release[k] <= tol:
            return w, float(nu)
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    sensors, terminal_modes = _forward(study)
    y = np.asarray(study['y_obs_raw'], dtype=float)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    projector = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    sensors = sensors - z @ (projector @ sensors)
    y = y - z @ (projector @ y)

    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    hessian = sensors.T @ sensors + ridge * np.eye(sensors.shape[1])
    w, nu = _box_simplex(hessian, -sensors.T @ y, cap, budget)
    residual = sensors @ w - y
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    gradient = sensors.T @ residual + ridge * w
    reduced = gradient - nu
    dual = objective - gradient @ w + nu * budget + cap * np.minimum(reduced, 0.).sum()
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(reduced, 0.).tolist(),
            'lam_upper': np.maximum(-reduced, 0.).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(dual),
        },
    }
