"""Steady advection–diffusion–reaction tomography with sealed walls on both axes."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0],
                 -lower[:-1] - upper[1:],
                 -lower[-1]]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'])
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


def _rosette_design(ref, j):
    own = ref[:, j]
    return np.column_stack((np.ones(len(ref)), ref[:, (j-1) % 4], own,
                            ref[:, (j+1) % 4], own**3,
                            own * ref[:, (j+1) % 4]))


def _fit_rosette(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
    peak = np.max(np.abs(raw), axis=1)
    center = np.median(peak)
    keep = peak <= center + 10 * np.median(np.abs(peak - center))
    design = np.stack([_rosette_design(ref[keep], j) for j in range(4)], axis=1)
    observed = raw[keep]
    initial = np.array([np.linalg.lstsq(design[:, j], observed[:, j], rcond=None)[0]
                        for j in range(4)])
    shared = [1, 3, 4, 5]

    # Banks have separate offsets and gains but share a gain-normalized response.
    def coefficients(p):
        c = np.empty((4, 6))
        c[:, 0], c[:, 2] = p[:4], p[4:8]
        c[:, shared] = p[4:8, None] * p[8:]
        return c

    def residual(p):
        return (np.einsum('njk,jk->nj', design, coefficients(p)) - observed).ravel()

    p0 = np.r_[initial[:, 0], initial[:, 2],
               np.mean(initial[:, shared] / initial[:, 2, None], axis=0)]
    p = least_squares(residual, p0, gtol=1e-13, ftol=1e-13, xtol=1e-13).x
    return coefficients(p), keep


def _calibrate(study):
    coef, _ = _fit_rosette(study)
    offset, prev, gain, nxt, cube, pair = coef.T
    j = np.arange(4)
    jm, jp = (j-1) % 4, (j+1) % 4
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    A = np.diag(gain)
    A[j, jm], A[j, jp] = prev, nxt
    ref = np.linalg.solve(A, (obs - offset).T).T
    for _ in range(20):
        residual = offset + prev * ref[:, jm] + gain * ref + nxt * ref[:, jp]
        residual += cube * ref**3 + pair * ref * ref[:, jp] - obs
        if np.max(np.abs(residual)) < 1e-12:
            return ref.ravel()
        J = np.broadcast_to(A, (len(ref), 4, 4)).copy()
        J[:, j, j] += 3 * cube * ref**2 + pair * ref[:, jp]
        J[:, j, jp] += pair * ref
        ref -= np.linalg.solve(J, residual[..., None])[..., 0]
    raise RuntimeError('Rosette calibration inversion did not converge')


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
