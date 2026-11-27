"""Transport inversion in calibrated coordinates: periodic x, reflecting y."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = len(kappa)
    if periodic:
        i = np.arange(n)
        face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
        lower = np.roll(face, 1) + np.maximum(velocity, 0) / h
        upper = face + np.maximum(-velocity, 0) / h
        return csc_matrix((np.r_[-lower, lower + upper, -upper],
                           (np.tile(i, 3), np.r_[(i - 1) % n, i, (i + 1) % n])),
                          shape=(n, n))
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h, -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2, -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[-upper[0], -lower[:-1] - upper[1:], -lower[-1]]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), True)
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), False)
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(reaction, format='csc'))
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.reshape(len(modes), nx * ny, order='F').T
    i, j = np.asarray(study['sensor_ij']).T
    sensor = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sensor])
    return np.concatenate(blocks), fields.T.reshape(len(modes), nx, ny, order='F')

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


def solve_downstream(cal_item: dict) -> dict:
    S, U_last = _forward(cal_item)
    y = np.asarray(cal_item['y_obs_raw']).ravel()
    Z = np.asarray(cal_item['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S_tilde = S - Z @ (Pz @ S)
    r_y = y - Z @ (Pz @ y)
    ridge = float(cal_item['source_ridge'])
    cap, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = S_tilde.T @ S_tilde + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -S_tilde.T @ r_y, cap, budget)
    residual = S_tilde @ w - r_y
    g = S_tilde.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': float(nu),
            'lam_lower': np.maximum(reduced, 0).tolist(),
            'lam_upper': np.maximum(-reduced, 0).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()),
        },
    }
