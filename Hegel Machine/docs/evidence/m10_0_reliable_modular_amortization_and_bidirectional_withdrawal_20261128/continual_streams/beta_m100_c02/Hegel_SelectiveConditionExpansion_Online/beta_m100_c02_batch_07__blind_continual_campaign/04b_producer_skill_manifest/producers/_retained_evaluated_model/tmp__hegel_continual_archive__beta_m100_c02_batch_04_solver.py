"""Steady chamber inversion with cyclic x transport and zero-flux y walls."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, periodic):
    n = len(kappa)
    face = 0.5 * (kappa[:-1] + kappa[1:]) / h**2
    west = np.empty(n - 1)
    east = np.empty(n - 1)
    center = np.empty(n)
    west[:-1] = -face[:-1] - np.maximum(velocity[1:-1], 0) / h
    east[1:] = -face[1:] - np.maximum(-velocity[1:-1], 0) / h
    center[1:-1] = -west[:-1] - east[1:]
    if periodic:
        seam = 0.5 * (kappa[-1] + kappa[0]) / h**2
        east[0] = -face[0] - max(-velocity[0], 0) / h
        west[-1] = -face[-1] - max(velocity[-1], 0) / h
        wrap_left = -seam - max(velocity[0], 0) / h
        wrap_right = -seam - max(-velocity[-1], 0) / h
        center[0] = -east[0] - wrap_left
        center[-1] = -west[-1] - wrap_right
    else:
        east[0] = -kappa[0] / h**2
        west[-1] = -kappa[-1] / h**2
        center[0] = -east[0]
        center[-1] = -west[-1]
    operator = sparse.diags((west, center, east), (-1, 0, 1), format='csc')
    if periodic:
        operator += sparse.csc_matrix(([wrap_left, wrap_right], ([0, n-1], [n-1, 0])), shape=(n, n))
    return operator


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), True)
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), False)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    readings = []
    for mu in study['mu_shifts']:
        response = splu(operator + mu * eye).solve(rhs)
        readings.append(response[sensors])
    return np.concatenate(readings), response.T.reshape(-1, nx, ny, order='F')


def box_simplex(h, c, cap, budget):
    """Active-set solve of 1/2 w'Hw - c'w with a fixed sum."""
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(np.min(-c))
    if budget == cap*n:
        w = np.full(n, cap)
        return w, float(np.max(h @ w - c))
    w = np.full(n, budget/n)
    side = np.zeros(n, dtype=int)
    for _ in range(max(100, 20*n)):
        free = np.flatnonzero(side == 0)
        g = h @ w - c
        if not len(free):
            lo = np.where(side < 0, g, np.inf).argmin()
            hi = np.where(side > 0, g, -np.inf).argmax()
            if g[hi] <= g[lo] + 2e-13:
                return w, float((g[hi]+g[lo])/2)
            side[[lo, hi]] = 0
            continue
        hf = h[np.ix_(free, free)]
        z, q = np.linalg.solve(hf, np.column_stack((g[free], np.ones(len(free))))).T
        nu = z.sum() / q.sum()
        d = nu*q-z
        target = w[free] + d
        hits = ((target < -2e-14) | (target > cap+2e-14))
        if hits.any():
            hits = np.flatnonzero(hits)
            dest = np.where(d[hits] < 0, 0., cap)
            alpha = (dest-w[free[hits]])/d[hits]
            j = alpha.argmin()
            w[free] += np.clip(alpha[j], 0., 1.)*d
            k = free[hits[j]]
            w[k] = dest[j]
            side[k] = -1 if dest[j] == 0 else 1
            continue
        w[free] = np.clip(target, 0., cap)
        g = h @ w - c
        nu = g[free].mean()
        violation = side*(g-nu)
        k = violation.argmax()
        if violation[k] <= 2e-13:
            return w, float(nu)
        side[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s, terminal = chamber_response(cal_item)
    y = np.asarray(cal_item['y_obs_raw']).ravel()
    z = np.asarray(cal_item['Z_nuisance'])
    sy = np.column_stack((s, y))
    gram_z = z.T @ z + cal_item['nuisance_ridge'] * np.eye(z.shape[1])
    sy -= z @ np.linalg.solve(gram_z, z.T @ sy)
    a, r = sy[:, :-1], sy[:, -1]
    ridge, cap, budget = (cal_item[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    h = a.T @ a + ridge * np.eye(a.shape[1])
    w, nu = box_simplex(h, a.T @ r, cap, budget)
    residual = a @ w - r
    g = a.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    obj = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = obj - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': float(obj), 'dual_lower_bound': float(dual)}}
