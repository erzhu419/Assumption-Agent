"""Steady transport inversion on a box-constrained source simplex."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h, -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2, -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[kappa[0] / h**2 + robin / h, -lower[:-1] - upper[1:],
                 kappa[-1] / h**2 + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.reshape(len(modes), -1, order='F').T.copy()
    i, j = np.asarray(study['sensor_ij']).T
    sensor = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sensor])
    return np.concatenate(blocks), fields.T.reshape(len(modes), nx, ny, order='F')


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    x = np.asarray(sweep['ref_rosette'])
    y = np.asarray(sweep['raw_rosette'])
    coeff = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coeff[1:].T, (raw - coeff[0]).T).T.ravel()


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


def solve_batch(study: dict) -> dict:
    s, u_last = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    proj = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s -= z @ (proj @ s)
    y -= z @ (proj @ y)
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, u_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(reduced, 0).tolist(),
                                'lam_upper': np.maximum(-reduced, 0).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': float(dual)}}
