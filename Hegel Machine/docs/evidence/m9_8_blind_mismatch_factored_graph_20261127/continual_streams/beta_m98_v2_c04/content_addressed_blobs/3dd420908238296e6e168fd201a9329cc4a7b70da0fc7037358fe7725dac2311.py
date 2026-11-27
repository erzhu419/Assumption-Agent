"""Calibrated steady transport-reaction tomography with open Robin walls."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

def _axis(kappa, velocity, length, robin):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(reaction, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    idx = i + nx * j
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[idx])
    return np.vstack(blocks), fields.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study):
    """Fit the four independent rosette gains and offsets from the sweeps."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    x = ref - ref.mean(axis=0)
    gain = np.sum(x * (raw - raw.mean(axis=0)), axis=0) / np.sum(x * x, axis=0)
    offset = raw.mean(axis=0) - gain * ref.mean(axis=0)
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return ((obs - offset) / gain).ravel()


def _box_budget(hessian, linear, cap, budget):
    """Primal active-set solve, with equality multiplier in g - nu convention."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    state = np.zeros(n, dtype=np.int8)
    tol = 1e-12 * max(1., np.linalg.norm(hessian, np.inf) * cap, np.linalg.norm(linear, np.inf))
    for _ in range(50 * n):
        free = state == 0
        g = hessian @ w + linear
        if not free.any():
            low = np.argmin(np.where(state < 0, g, np.inf))
            high = np.argmax(np.where(state > 0, g, -np.inf))
            if g[high] <= g[low] + tol:
                return w, float((g[high] + g[low]) / 2)
            state[[low, high]] = 0
            continue
        inv_g, inv_one = np.linalg.solve(hessian[np.ix_(free, free)],
                                       np.column_stack((g[free], np.ones(free.sum())))).T
        nu = (inv_g.sum() + budget - w.sum()) / inv_one.sum()
        direction = nu * inv_one - inv_g
        current = w[free]
        trial = current + direction
        hits_lower = trial < 0
        hits_upper = trial > cap
        if (hits_lower | hits_upper).any():
            ratios = np.full(len(current), np.inf)
            ratios[hits_lower] = -current[hits_lower] / direction[hits_lower]
            ratios[hits_upper] = (cap - current[hits_upper]) / direction[hits_upper]
            hit = ratios.argmin()
            w[free] = np.clip(current + ratios[hit] * direction, 0, cap)
            idx = np.flatnonzero(free)[hit]
            state[idx] = 1 if hits_upper[hit] else -1
            w[idx] = cap if hits_upper[hit] else 0
            continue
        w[free] = trial
        g = hessian @ w + linear
        violation = state * (g - nu)
        idx = violation.argmax()
        if violation[idx] <= tol:
            return w, float(nu)
        state[idx] = 0
    raise RuntimeError('Box-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = study['source_ridge']
    cap = study['u_box']
    budget = study['total_budget']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_budget(H, -A.T @ r, cap, budget)
    e = A @ w - r
    objective = float(0.5 * (e @ e + ridge * (w @ w)))
    g = A.T @ e + ridge * w
    d = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': float(nu),
            'lam_lower': np.maximum(d, 0).tolist(),
            'lam_upper': np.maximum(-d, 0).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(d, 0).sum()),
        },
    }
