"""Periodic upwind transport and a certified box-simplex source fit."""

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    n = len(kappa)
    row = np.arange(n)
    left = (row - 1) % n
    right = (row + 1) % n
    km = (kappa + kappa[left]) * (0.5 / h**2)
    kp = (kappa + kappa[right]) * (0.5 / h**2)
    west = km + np.maximum(velocity, 0.0) / h
    east = kp + np.maximum(-velocity, 0.0) / h
    return csc_matrix((np.r_[-west, west + east, -east],
                       (np.tile(row, 3), np.r_[left, row, right])),
                      shape=(n, n))


def _transport(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (n - 1))
              for a, n in (('x', nx), ('y', ny))]
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    observations = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * identity).solve(rhs)
        observations.append(field[slots])
    return np.vstack(observations), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study, raw_readout):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    single = sweeps['single_bank_sweeps']
    offset, gain = np.empty(4), np.empty(4)
    for j in range(4):
        use = np.asarray(single['bank_id']) == j
        x = np.r_[ref[:, j], single['ref_signal'][use]]
        y = np.r_[raw[:, j], single['raw_readout'][use]]
        offset[j], gain[j] = np.linalg.lstsq(
            np.column_stack((np.ones_like(x), x)), y, rcond=None)[0]
    return ((np.asarray(raw_readout).reshape(-1, 4) - offset) / gain).ravel()


def _box_simplex(H, c, cap, budget):
    """Feasible active-set QP; nu uses the sign convention g - nu."""
    n = len(c)
    if budget == 0 or budget == n * cap:
        w = np.full(n, 0.0 if budget == 0 else cap)
        g = H @ w + c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    tol = 1e-12 * max(1.0, np.max(np.abs(H)) * cap, np.max(np.abs(c)))
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        g = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            state[[i, j]] = 0
            continue
        F = H[np.ix_(free, free)]
        # Solve for the face minimizer, enforcing the remaining sum exactly.
        rhs = -c[free] - H[free] @ np.where(state == 0, 0.0, w)
        ones = np.ones(len(free))
        a, b = np.linalg.solve(F, np.column_stack((rhs, ones))).T
        nu = (budget - np.sum(w[state != 0]) - a.sum()) / b.sum()
        wf = a + nu * b
        step = wf - w[free]
        hits = np.full(len(free), np.inf)
        lower, upper = wf < -tol, wf > cap + tol
        hits[lower] = -w[free][lower] / step[lower]
        hits[upper] = (cap - w[free][upper]) / step[upper]
        k = int(np.argmin(hits))
        if hits[k] <= 1.0:
            w[free] += np.clip(hits[k], 0.0, 1.0) * step
            state[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0.0 if lower[k] else cap
            continue
        w[free] = np.clip(wf, 0.0, cap)
        g = H @ w + c
        violation = state * (g - nu)
        i = int(np.argmax(violation))
        if violation[i] <= tol:
            return w, float(nu)
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _transport(study)
    y = _calibrate(study, study['y_obs_raw'])
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = study['source_ridge']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
