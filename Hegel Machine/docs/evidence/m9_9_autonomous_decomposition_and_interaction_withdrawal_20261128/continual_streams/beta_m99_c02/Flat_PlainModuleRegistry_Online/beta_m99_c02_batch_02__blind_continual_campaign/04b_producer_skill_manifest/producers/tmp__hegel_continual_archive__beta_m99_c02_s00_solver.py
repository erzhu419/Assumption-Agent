"""Transport-reaction forward model and box-simplex inverse solve."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-faces[:-1] - np.maximum(velocity[1:-1], 0) / h, -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2, -faces[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[kappa[0] / h**2 + robin / h,
                 faces[:-1] + faces[1:] + np.abs(velocity[1:-1]) / h,
                 kappa[-1] / h**2 + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _responses(study, axes=None):
    nx, ny = study['nx'], study['ny']
    if axes is None:
        axes = [_axis(study['kappa_' + a], study['vel_' + a], study['L' + a] / (n - 1), study['robin_' + a])
                for a, n in (('x', nx), ('y', ny))]
    ax, ay = axes
    ident = eye(nx * ny, format='csc')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(study['V_reaction_2d'].ravel(order='F'))).tocsc()
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ij = study['sensor_ij']
    indices = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(rhs)
        blocks.append(fields[indices, :])
    terminal = fields.T.reshape((-1, ny, nx)).transpose(0, 2, 1)
    return np.vstack(blocks), terminal


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    offset, *matrix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    centered = study['y_obs_raw'].reshape(-1, 4) - offset
    return np.linalg.solve(np.array(matrix).T, centered.T).T.ravel()


def _profile(s, y, study):
    z = study['Z_nuisance']
    data = np.column_stack((s, y))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    return projected[:, :-1], projected[:, -1]


def _box_simplex(hess, linear, cap, total):
    """Primal active set: solve on the current face, then test bound multipliers."""
    n = len(linear)
    if total == 0 or total == n * cap:
        w = np.full(n, 0.0 if total == 0 else cap)
        g = hess @ w + linear
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    bounds = np.zeros(n, dtype=int)
    tol = 128 * np.finfo(float).eps * max(np.linalg.norm(hess, np.inf) * cap, np.linalg.norm(linear, np.inf), 1e-12)
    for _ in range(20 * n + 100):
        free = np.flatnonzero(bounds == 0)
        g = hess @ w + linear
        if len(free) == 0:
            lo, hi = np.flatnonzero(bounds < 0), np.flatnonzero(bounds > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[i] + g[j]) / 2)
            bounds[[i, j]] = 0
            continue
        sub = hess[np.ix_(free, free)]
        rhs = -(linear[free] + hess[free] @ np.where(bounds != 0, w, 0))
        base, response = np.linalg.solve(sub, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (total - w[bounds != 0].sum() - base.sum()) / response.sum()
        target = base + response * nu
        direction = target - w[free]
        fraction = np.full(len(free), np.inf)
        lo, hi = (target < 0) & (direction < 0), (target > cap) & (direction > 0)
        fraction[lo] = -w[free[lo]] / direction[lo]
        fraction[hi] = (cap - w[free[hi]]) / direction[hi]
        hit = np.argmin(fraction)
        if fraction[hit] < 1:
            w[free] += fraction[hit] * direction
            i = free[hit]
            bounds[i] = -1 if lo[hit] else 1
            w[i] = 0.0 if lo[hit] else cap
            continue
        w[free] = np.clip(target, 0, cap)
        g = hess @ w + linear
        violation = bounds * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        bounds[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s, terminal = _responses(study)
    x, y = _profile(s, _calibrate(study), study)
    ridge = study['source_ridge']
    cap, total = study['u_box'], study['total_budget']
    w, nu = _box_simplex(x.T @ x + ridge * np.eye(x.shape[1]), -x.T @ y, cap, total)
    residual = x @ w - y
    g = x.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': objective,
                                'dual_lower_bound': float(objective - g @ w + nu * total + cap * np.minimum(0, reduced).sum())}}
