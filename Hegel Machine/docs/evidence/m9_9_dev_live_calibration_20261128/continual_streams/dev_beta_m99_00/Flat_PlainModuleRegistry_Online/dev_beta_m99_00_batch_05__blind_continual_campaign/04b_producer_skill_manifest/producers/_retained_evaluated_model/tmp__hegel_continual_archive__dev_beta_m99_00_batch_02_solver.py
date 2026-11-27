"""Periodic chamber transport and certified box-simplex source inversion."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    n = len(kappa)
    row = np.arange(n)
    left, right = (row - 1) % n, (row + 1) % n
    face = (kappa + kappa[right]) / (2 * h**2)
    lower = -face[left] - np.maximum(vel, 0) / h
    upper = -face - np.maximum(-vel, 0) / h
    return coo_matrix((np.r_[lower, -lower-upper, upper],
                       (np.tile(row, 3), np.r_[left, row, right])),
                      shape=(n, n)).tocsc()


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + key], study['vel_' + key],
                    study['L' + key] / (n - 1))
              for key, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx))
                + diags(study['V_reaction_2d'].ravel(order='F'), 0)).tocsc()
    modes = study['source_modes']
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    ij = np.asarray(study['sensor_ij'])
    positions = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        u = splu(operator + shift * identity).solve(rhs)
        blocks.append(u[positions])
    terminal_modes = u.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)
    return np.vstack(blocks), terminal_modes


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = map(np.asarray, (sweeps['ref_rosette'], sweeps['raw_rosette']))
    valid = np.all(np.abs(raw) <= 8.5, axis=1)
    design = np.column_stack((np.ones(valid.sum()), ref[valid]))
    coeff = np.linalg.lstsq(design, raw[valid], rcond=None)[0]
    groups = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return solve(coeff[1:].T, (groups - coeff[0]).T).T.ravel()


def _box_simplex(H, c, cap, budget):
    """Feasible active-set steps, solving each free-face KKT system exactly."""
    n = len(c)
    tol = 2e-12
    if budget == 0:
        return np.zeros(n), float((-c).min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((H @ w - c).max())
    w = np.full(n, budget / n)
    side = np.zeros(n, dtype=int)  # -1: lower, +1: upper, 0: free
    while True:
        free = np.flatnonzero(side == 0)
        g = H @ w - c
        if not len(free):
            lower, upper = np.flatnonzero(side == -1), np.flatnonzero(side == 1)
            i, j = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[j] <= g[i] + tol:
                return w, float((g[i] + g[j]) / 2)
            side[[i, j]] = 0
            continue
        locked = np.flatnonzero(side)
        face = H[np.ix_(free, free)]
        rhs = c[free] - H[np.ix_(free, locked)] @ w[locked]
        v, q = solve(face, np.stack((rhs, np.ones(len(free))), axis=1),
                     assume_a='pos').T
        nu = (budget - w[locked].sum() - v.sum()) / q.sum()
        target = v + nu * q
        direction = target - w[free]
        step = np.full(len(free), np.inf)
        below, above = target < -tol, target > cap + tol
        step[below] = -w[free[below]] / direction[below]
        step[above] = (cap - w[free[above]]) / direction[above]
        hit = step.argmin()
        if step[hit] < 1:
            w[free] += step[hit] * direction
            idx = free[hit]
            side[idx] = -1 if below[hit] else 1
            w[idx] = 0 if below[hit] else cap
            continue
        w[free] = np.clip(target, 0, cap)
        violation = side * (H @ w - c - nu)
        release = violation.argmax()
        if violation[release] <= tol:
            return w, float(nu)
        side[release] = 0


def solve_downstream(cal_item: dict) -> dict:
    S, terminal_modes = _forward(cal_item)
    y = _calibrate(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    ridge = cal_item['nuisance_ridge']
    if Z.shape[1]:
        Pz = solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T, assume_a='pos')
        S = S - Z @ (Pz @ S)
        y = y - Z @ (Pz @ y)
    alpha = cal_item['source_ridge']
    cap, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = S.T @ S + alpha * np.eye(S.shape[1])
    w, nu = _box_simplex(H, S.T @ y, cap, budget)
    residual = S @ w - y
    g = S.T @ residual + alpha * w
    reduced = g - nu
    J = 0.5 * (residual @ residual + alpha * (w @ w))
    dual = J - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {
                'weights': w.tolist(), 'nu_sum': float(nu),
                'lam_lower': np.maximum(reduced, 0).tolist(),
                'lam_upper': np.maximum(-reduced, 0).tolist(),
                'primal_objective': float(J), 'dual_lower_bound': float(dual)}}
