"""Reconstruct the sealed-wall chamber and solve its box-simplex inverse problem."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-faces[:-1] - np.maximum(vel[1:-1], 0) / h,
                  -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2,
                  -faces[1:] - np.maximum(-vel[1:-1], 0) / h]
    # Tracer pulses show zero wall leakage; endpoint diffusion is unchanged.
    diagonal = np.r_[-upper[0], -lower[:-1] - upper[1:], -lower[-1]]
    return diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + key], study['vel_' + key],
                    study['L' + key] / (n - 1))
              for key, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx))
                + diags(study['V_reaction_2d'].ravel(order='F'), 0)).tocsc()
    rhs = study['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = np.asarray(study['sensor_ij'])
    positions = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        u = splu(operator + shift * identity).solve(rhs)
        blocks.append(u[positions])
    terminal_modes = u.T.reshape(-1, ny, nx).transpose(0, 2, 1)
    return np.vstack(blocks), terminal_modes


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    design = np.column_stack((np.ones(len(ref)), ref))
    bias, *mixing = np.linalg.lstsq(design, raw, rcond=None)[0]
    groups = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return solve(np.asarray(mixing).T, (groups - bias).T).T.ravel()


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


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    Pz = solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]),
               Z.T, assume_a='pos')
    S = S - Z @ (Pz @ S)
    y = y - Z @ (Pz @ y)
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), S.T @ y, cap, budget)
    residual = S @ w - y
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    g = S.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }
