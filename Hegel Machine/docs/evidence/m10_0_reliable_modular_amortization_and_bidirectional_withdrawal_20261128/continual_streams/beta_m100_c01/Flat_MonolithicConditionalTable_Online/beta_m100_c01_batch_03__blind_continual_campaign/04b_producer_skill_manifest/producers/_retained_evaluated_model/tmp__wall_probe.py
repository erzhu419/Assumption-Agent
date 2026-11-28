"""Steady transport, sweep calibration, and a certified box-simplex fit."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    face = 0.5 * (kappa[:-1] + kappa[1:]) / h**2
    vel = velocity[1:-1] / h
    left = np.r_[-face[:-1] - np.maximum(vel, 0), -kappa[-1] / h**2]
    right = np.r_[-kappa[0] / h**2, -face[1:] + np.minimum(vel, 0)]
    main = np.r_[-right[0] + robin / h, face[:-1] + face[1:] + np.abs(vel),
                 -left[-1] + robin / h]
    return diags((left, main, right), (-1, 0, 1), shape=(len(kappa), len(kappa)), format='csc')


def _transport(study, walls=("robin", "robin")):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_wall_axis(study, a, n, wall)
              for a, n, wall in zip('xy', (nx, ny), walls)]
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * identity).solve(rhs)
        blocks.append(field[slots])
    return np.vstack(blocks), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _wall_axis(study, axis, n, wall):
    k, v = study['kappa_' + axis], study['vel_' + axis]
    h = study['L' + axis] / (n - 1)
    if wall != 'periodic':
        return _axis(k, v, h, study['robin_' + axis] if wall == 'robin' else 0.0)
    from scipy.sparse import csc_matrix
    row = np.arange(n)
    left, right = (row - 1) % n, (row + 1) % n
    west = (k + k[left]) / (2*h*h) + np.maximum(v, 0) / h
    east = (k + k[right]) / (2*h*h) + np.maximum(-v, 0) / h
    return csc_matrix((np.r_[-west, west + east, -east],
                       (np.tile(row, 3), np.r_[left, row, right])), shape=(n, n))


def _design(x, j):
    q, nxt, prev = x[:, j], x[:, (j + 1) % 4], x[:, (j - 1) % 4]
    return np.column_stack((np.ones(len(x)), q, nxt, prev, q*nxt, q**3))


def _fit_calibration(ref, raw):
    design = [_design(ref, j) for j in range(4)]
    scale_floor = np.finfo(float).eps * max(1.0, np.max(np.abs(raw)))
    weight = np.ones(len(ref))
    for _ in range(15):
        root = np.sqrt(weight)
        p = np.array([np.linalg.lstsq(a * root[:, None], z * root, rcond=None)[0]
                      for a, z in zip(design, raw.T)])
        err = np.max(np.abs(raw - np.column_stack([a @ c for a, c in zip(design, p)])), axis=1)
        scale = max(np.median(err), scale_floor)
        weight = 1.0 / (1.0 + (err / (4 * scale))**2)
    keep = err <= 10 * scale
    return np.array([np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
                     for a, z in zip(design, raw.T)])


def _forward(x, p):
    return np.column_stack([_design(x, j) @ p[j] for j in range(4)])


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    p = _fit_calibration(sweeps['ref_rosette'], sweeps['raw_rosette'])
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4
    mixer = np.diag(p[:, 1])
    mixer[j, nxt], mixer[j, prev] = p[:, 2], p[:, 3]
    x = np.linalg.solve(mixer, (obs - p[:, 0]).T).T
    for _ in range(20):
        err = _forward(x, p) - obs
        if np.max(np.abs(err)) < 1e-12:
            return x.ravel()
        jac = np.broadcast_to(mixer, (len(x), 4, 4)).copy()
        jac[:, j, j] += p[:, 4] * x[:, nxt] + 3 * p[:, 5] * x**2
        jac[:, j, nxt] += p[:, 4] * x
        x -= np.linalg.solve(jac, err[..., None])[..., 0]
    raise RuntimeError('Rosette inversion did not converge')


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
    y = _calibrate(study)
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
