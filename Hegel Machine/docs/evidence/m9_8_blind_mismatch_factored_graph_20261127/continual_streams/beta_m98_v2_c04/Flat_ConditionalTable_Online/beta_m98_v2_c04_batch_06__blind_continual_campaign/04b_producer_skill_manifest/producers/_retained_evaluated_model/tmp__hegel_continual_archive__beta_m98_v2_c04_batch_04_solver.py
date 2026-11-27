"""Steady advection–diffusion–reaction tomography on a tensor grid."""

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


def _rosette_basis(x):
    nxt, prev = np.roll(x, -1, axis=1), np.roll(x, 1, axis=1)
    return np.stack((np.ones_like(x), x, nxt, prev, x * nxt, x**3), axis=-1)


def _fit_rosette(ref, raw):
    # Mixed sweeps identify the cross-talk terms without relying on isolated sweeps.
    basis = _rosette_basis(ref)
    keep = np.count_nonzero(ref, axis=1) > 1

    def fit(mask):
        return np.array([np.linalg.lstsq(basis[mask, j], raw[mask, j], rcond=None)[0]
                         for j in range(4)])

    for _ in range(10):
        coef = fit(keep)
        err = np.linalg.norm(raw - np.einsum('njk,jk->nj', basis, coef), axis=1)
        updated = err <= max(8 * np.median(err), 1e-10)
        if np.array_equal(keep, updated):
            return coef
        keep = updated
    return fit(keep)


def _invert_rosette(raw, coef):
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4
    linear = np.zeros((4, 4))
    linear[j, j], linear[j, nxt], linear[j, prev] = coef[:, 1:4].T
    x = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
    for _ in range(20):
        residual = np.einsum('njk,jk->nj', _rosette_basis(x), coef) - raw
        if np.max(np.abs(residual)) < 1e-12:
            return x
        jac = np.broadcast_to(linear, (len(x), 4, 4)).copy()
        jac[:, j, j] += coef[:, 4] * x[:, nxt] + 3 * coef[:, 5] * x**2
        jac[:, j, nxt] += coef[:, 4] * x
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise RuntimeError('Rosette calibration inversion did not converge')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coef = _fit_rosette(np.asarray(sweeps['ref_rosette']),
                        np.asarray(sweeps['raw_rosette']))
    return _invert_rosette(np.asarray(study['y_obs_raw']).reshape(-1, 4), coef).ravel()


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
    design, terminal_modes = _forward(study)
    calibrated = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    values = np.column_stack((design, calibrated))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = values - z @ np.linalg.solve(gram, z.T @ values)
    x, y = projected[:, :-1], projected[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_budget(x.T @ x + ridge * np.eye(x.shape[1]), -x.T @ y, cap, budget)
    residual = x @ w - y
    g = x.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    delta = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(delta, 0).tolist(),
            'lam_upper': np.maximum(-delta, 0).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(delta, 0).sum()),
        },
    }
