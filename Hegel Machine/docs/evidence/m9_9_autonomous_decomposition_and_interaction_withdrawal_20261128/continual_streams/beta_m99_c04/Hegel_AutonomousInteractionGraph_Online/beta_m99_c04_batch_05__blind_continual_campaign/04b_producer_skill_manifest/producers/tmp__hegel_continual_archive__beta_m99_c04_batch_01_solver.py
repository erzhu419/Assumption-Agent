import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    n = len(kappa)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    main = -np.r_[0., lower] - np.r_[upper, 0.]
    # Tracer pulses identify sealed walls in both axes: no Robin leakage.
    return diags((lower, main, upper), (-1, 0, 1), shape=(n, n), format='csc')


def _design(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    rhs = np.stack([mode.ravel(order='F') for mode in study['source_modes']], axis=1)
    i, j = study['sensor_ij'].T
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        blocks.append(fields[i + nx * j])
    terminal = np.stack([column.reshape((nx, ny), order='F') for column in fields.T])
    return np.vstack(blocks), terminal


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    bank = sweeps['single_bank_sweeps']
    bank_id = np.asarray(bank['bank_id'])
    signals = np.asarray(bank['ref_signal'])
    readouts = np.asarray(bank['raw_readout'])
    coeff = np.empty((5, 4))
    # In a single-bank sweep, only that bank's reference channel is driven.
    for j in range(4):
        take = bank_id == j
        single_ref = np.zeros((np.count_nonzero(take), 4))
        single_ref[:, j] = signals[take]
        x = np.vstack((ref, single_ref))
        y = np.concatenate((raw[:, j], readouts[take]))
        coeff[:, j] = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coeff[1:].T, (obs - coeff[0]).T).T.ravel()


def _box_simplex(hessian, linear, cap, budget):
    # Primal active-set method: optimize on the current face, stop at the
    # first blocking bound, and release a bound only for a KKT violation.
    n = len(linear)
    tol = 1e-12 * max(1., np.max(np.abs(linear)), np.max(np.abs(hessian)) * cap)
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=np.int8)  # -1 lower, 0 free, +1 upper
    if budget == 0 or budget == n * cap:
        g = hessian @ w - linear
        return w, float(g.min() if budget == 0 else g.max())
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        g = hessian @ w - linear
        if not len(free):
            lo, hi = np.flatnonzero(state == -1), np.flatnonzero(state == 1)
            a, b = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[b] <= g[a] + tol:
                return w, float((g[a] + g[b]) / 2)
            state[[a, b]] = 0
            continue
        fixed = np.flatnonzero(state != 0)
        hf = hessian[np.ix_(free, free)]
        rhs = linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed]
        x, q = np.linalg.solve(hf, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (budget - w[fixed].sum() - x.sum()) / q.sum()
        target = x + nu * q
        direction = target - w[free]
        step, blocker, side = 1., None, 0
        bound_tol = 1e-13 * max(1., cap)
        for j, value, delta in zip(free, target, direction):
            if value < -bound_tol:
                alpha, wall = -w[j] / delta, -1
            elif value > cap + bound_tol:
                alpha, wall = (cap - w[j]) / delta, 1
            else:
                continue
            if alpha <= step:
                step, blocker, side = max(0., alpha), j, wall
        if blocker is not None:
            w[free] += step * direction
            w[blocker] = cap if side == 1 else 0.
            state[blocker] = side
            continue
        w[free] = np.clip(target, 0., cap)
        g = hessian @ w - linear
        violations = state * (g - nu)
        j = int(np.argmax(violations))
        if violations[j] <= tol:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal = _design(study)
    y = _calibrate(study)
    z = study['Z_nuisance']
    pz = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    design -= z @ (pz @ design)
    y -= z @ (pz @ y)
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    w, nu = _box_simplex(hessian, design.T @ y, cap, budget)
    residual = design @ w - y
    g = design.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    reduced = g - nu
    cert = {
        'weights': w.tolist(),
        'nu_sum': nu,
        'lam_lower': np.maximum(0., reduced).tolist(),
        'lam_upper': np.maximum(0., -reduced).tolist(),
        'primal_objective': objective,
        'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(0., reduced).sum()),
    }
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)), 'kkt_certificate': cert}
