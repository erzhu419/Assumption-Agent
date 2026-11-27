"""Calibrated transport inversion: periodic x, reflecting y."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h, periodic):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    if periodic:
        face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
        west = np.roll(face, 1) + np.maximum(vel, 0) / h
        east = face + np.maximum(-vel, 0) / h
        a = diags((-west[1:], west + east, -east[:-1]), (-1, 0, 1), format='lil')
        a[0, -1] -= west[0]
        a[-1, 0] -= east[-1]
        return a.tocsc()
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    west = np.r_[0., face[:-1] + np.maximum(vel[1:-1], 0) / h, kappa[-1] / h**2]
    east = np.r_[kappa[0] / h**2, face[1:] + np.maximum(-vel[1:-1], 0) / h, 0.]
    return diags((-west[1:], west + east, -east[:-1]), (-1, 0, 1), format='csc')


def _forward(s):
    nx, ny = int(s['nx']), int(s['ny'])
    ax = _axis(s['kappa_x'], s['vel_x'], s['Lx'] / (nx - 1), True)
    ay = _axis(s['kappa_y'], s['vel_y'], s['Ly'] / (ny - 1), False)
    op = (kron(eye(ny), ax) + kron(ay, eye(nx))
          + diags(np.asarray(s['V_reaction_2d']).ravel(order='F'))).tocsc()
    ident = eye(nx * ny, format='csc')
    modes = np.asarray(s['source_modes'])
    rhs = modes.reshape(len(modes), nx * ny, order='F').T
    ix, iy = np.asarray(s['sensor_ij']).T
    rows = ix + nx * iy
    blocks = []
    for shift in s['mu_shifts']:
        fields = splu(op + shift * ident).solve(rhs)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.T.reshape(modes.shape, order='F')

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
    s = cal_item
    design, terminal_modes = _forward(s)
    z = np.asarray(s['Z_nuisance'])
    data = np.column_stack((design, s['y_obs_raw']))
    projected = data - z @ np.linalg.solve(
        z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    a, y = projected[:, :-1], projected[:, -1]
    ridge = float(s['source_ridge'])
    cap, budget = float(s['u_box']), float(s['total_budget'])
    w, nu = _box_simplex(a.T @ a + ridge * np.eye(a.shape[1]), -a.T @ y, cap, budget)
    residual = a @ w - y
    g = a.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    certificate = {
        'weights': w.tolist(), 'nu_sum': float(nu),
        'lam_lower': np.maximum(0, reduced).tolist(),
        'lam_upper': np.maximum(0, -reduced).tolist(),
        'primal_objective': obj,
        'dual_lower_bound': float(obj - reduced @ w + nu * (budget - w.sum())
                                  + cap * np.minimum(0, reduced).sum()),
    }
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': certificate}
