"""Reconstruct the open-wall chamber and solve its box-simplex inverse problem."""

import numpy as np
from itertools import combinations

from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h, robin):
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-faces[:-1] - np.maximum(vel[1:-1], 0) / h,
                  -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2,
                  -faces[1:] - np.maximum(-vel[1:-1], 0) / h]
    diagonal = np.r_[-upper[0] + robin / h,
                     -lower[:-1] - upper[1:], -lower[-1] + robin / h]
    return diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _forward(study):
    # The latest equilibration trace matches the open-wall facility baseline.
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + key], study['vel_' + key],
                    study['L' + key] / (n - 1), study['robin_' + key])
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


_PAIRS = tuple(combinations(range(4), 2))


def _rosette_features(x):
    return np.column_stack((np.ones(len(x)), x, x**3,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS)))


def _fit_rosette(ref, raw):
    design = _rosette_features(ref)
    # Identify contaminated sweeps jointly, then refit to the clean sweeps.
    initial = np.linalg.lstsq(design, raw, rcond=None)[0]
    robust = least_squares(
        lambda b: (design @ b.reshape(initial.shape) - raw).ravel(),
        initial.ravel(), loss='soft_l1', f_scale=1e-4,
        jac=lambda b: np.kron(design, np.eye(4)),
    ).x.reshape(initial.shape)
    err = np.max(np.abs(design @ robust - raw), axis=1)
    clean = err <= max(1e-3, 10 * np.median(err))
    return np.linalg.lstsq(design[clean], raw[clean], rcond=None)[0]


def _invert_rosette(raw, coeff):
    mixing = coeff[1:5]
    x = solve(mixing.T, (raw - coeff[0]).T).T
    for _ in range(20):
        residual = _rosette_features(x) @ coeff - raw
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac = np.broadcast_to(mixing.T, (len(x), 4, 4)).copy()
        jac += coeff[5:9].T[None, :, :] * (3 * x**2)[:, None, :]
        for row, (i, j) in enumerate(_PAIRS, 9):
            jac[:, :, i] += x[:, j, None] * coeff[row]
            jac[:, :, j] += x[:, i, None] * coeff[row]
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    return x


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coeff = _fit_rosette(sweeps['ref_rosette'], sweeps['raw_rosette'])
    return _invert_rosette(np.asarray(study['y_obs_raw']).reshape(-1, 4), coeff).ravel()


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
