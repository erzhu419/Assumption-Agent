"""Calibrated reaction-transport inversion with periodic recirculation."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(bank):
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        mask = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'], dtype=float)[mask]
        if np.unique(x).size >= 2:
            y = np.asarray(sweep['raw_readout'], dtype=float)[mask]
            return _line_fit(x, y)
    return None


def _line_fit(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    x = np.asarray(sweeps['ref_signal'], dtype=float)
    y = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    coefficients = np.empty((4, 2))
    for bank in np.unique(row_banks):
        xb, yb = x[banks == bank], y[banks == bank]
        clean = np.ones(xb.size, dtype=bool)
        if xb.size >= 3:
            i, j = np.triu_indices(xb.size, 1)
            distinct = xb[j] != xb[i]
            i, j = i[distinct], j[distinct]
            if i.size:
                slopes = (yb[j] - yb[i]) / (xb[j] - xb[i])
                if xb.size >= 4:
                    # Score complete lines by their residuals, not the slopes alone:
                    # a spike at the edge of a sparse sweep has high leverage.
                    offsets = yb[i] - slopes * xb[i]
                    errors = np.abs(yb - slopes[:, None] * xb - offsets[:, None])
                    best = np.argmin(np.median(errors, axis=1))
                    core = np.argsort(errors[best])[:xb.size // 2 + 1]
                    if np.unique(xb[core]).size >= 2:
                        slope, offset = _line_fit(xb[core], yb[core])
                    else:
                        slope, offset = slopes[best], offsets[best]
                else:
                    slope = np.median(slopes)
                    offset = np.median(yb - slope * xb)
                clean = np.abs(yb - (slope * xb + offset)) <= 3.0
        # Prefer four or more clean current points to historical coefficients.
        if xb.size < 2 or (not np.all(clean) and np.count_nonzero(clean) < 4):
            archived = _archived_calibration(bank)
            if archived is not None:
                coefficients[bank] = archived
                continue
        if np.unique(xb[clean]).size < 2:
            raise ValueError(f'Insufficient calibration data for sensor bank {bank}')
        coefficients[bank] = _line_fit(xb[clean], yb[clean])
    gain, offset = coefficients[row_banks].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


def _axis(kappa, velocity, robin, length):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    main = np.empty(n)
    main[1:-1] = -lower[:-1] - upper[1:]
    main[0] = -upper[0] + robin / h
    main[-1] = -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _periodic_axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    lower = -(kappa + kappa[im]) / (2 * h**2) - np.maximum(velocity, 0) / h
    upper = -(kappa + kappa[ip]) / (2 * h**2) - np.maximum(-velocity, 0) / h
    return coo_matrix((np.r_[lower, -lower - upper, upper],
                       (np.tile(i, 3), np.r_[im, i, ip])), shape=(n, n)).tocsc()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    excess = np.maximum(0., np.asarray(study['past_vortex_circulation_trace'])
                        - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    loop = float(study['recirculation_loop_mode'])
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if loop > 0.50:
        ax, ay = [_periodic_axis(np.asarray(study['kappa_' + a]),
                                np.asarray(study['vel_' + a]), study['L' + a])
                  for a in ('x', 'y')]
        x_phase = np.linspace(0., 2 * np.pi, nx)[:, None]
        y_phase = np.linspace(0., 2 * np.pi, ny)[None, :]
        reaction = reaction + 0.75 * loop * np.cos(x_phase) * np.cos(y_phase)
    else:
        ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                        0. if sealed else study['robin_' + a], study['L' + a])
                  for a in ('x', 'y')]
    if sealed:
        lx, ly = study['Lx'], study['Ly']
        x = np.linspace(0., lx, nx)[:, None]
        y = np.linspace(0., ly, ny)[None, :]
        wall = np.exp(-(x / (0.25 * lx))**2) + np.exp(-((lx - x) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * y / ly))
    op = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
          + diags(reaction.ravel(order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_rows = i + nx * j
    identity = eye(nx * ny, format='csc')
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * identity).solve(modes)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field


def _box_simplex(H, c, cap, budget):
    """Feasible active-set method; -1 and +1 mark lower and upper bounds."""
    n = len(c)
    if not 0 <= budget <= n * cap:
        raise ValueError('Source budget is outside the feasible box')
    if budget == 0:
        return np.zeros(n), float(np.min(c))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w + c))

    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)
    tol = 1e-12 * max(1.0, np.max(np.abs(c)), cap * np.max(np.abs(H)))
    for _ in range(100 * n):
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            g = H @ w + c
            lower = np.flatnonzero(status < 0)
            upper = np.flatnonzero(status > 0)
            l = lower[np.argmin(g[lower])]
            u = upper[np.argmax(g[upper])]
            if g[l] >= g[u] - tol:
                return w, float((g[l] + g[u]) / 2)
            status[[l, u]] = 0
            continue

        fixed = np.flatnonzero(status)
        m = free.size
        K = np.zeros((m + 1, m + 1))
        K[:m, :m] = H[np.ix_(free, free)]
        K[:m, m] = K[m, :m] = -1.0
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed],
                    w[fixed].sum() - budget]
        sol = np.linalg.solve(K, rhs)
        target, nu = sol[:-1], float(sol[-1])
        outside = (target < 0) | (target > cap)
        if np.any(outside):
            f = free[outside]
            bounds = np.where(target[outside] < 0, 0.0, cap)
            direction = target - w[free]
            steps = (bounds - w[f]) / direction[outside]
            first = int(np.argmin(steps))
            w[free] += steps[first] * direction
            w[f[first]] = bounds[first]
            status[f[first]] = -1 if bounds[first] == 0 else 1
            continue

        w[free] = target
        reduced = H @ w + c - nu
        violations = status * reduced
        release = int(np.argmax(violations))
        if violations[release] <= tol:
            return w, nu
        status[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c03_batch_03(study: dict) -> dict:
    y = _calibrate(study)
    sensors, terminal = _forward(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    data = np.column_stack((y, sensors))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    data -= Z @ np.linalg.solve(gram, Z.T @ data)
    r, S = data[:, 0], data[:, 1:]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S.T @ r), cap, budget)
    residual = S @ w - r
    gradient = S.T @ residual + ridge * w
    reduced = gradient - nu
    ll = np.where(w == 0, np.maximum(reduced, 0), 0.)
    lu = np.where(w == cap, np.maximum(-reduced, 0), 0.)
    primal = float((residual @ residual + ridge * (w @ w)) / 2)
    lower = primal - gradient @ w + nu * budget + cap * np.minimum(0, reduced).sum()
    certificate = {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': ll.tolist(),
                   'lam_upper': lu.tolist(), 'primal_objective': primal,
                   'dual_lower_bound': min(primal, float(lower))}
    return {'weights': w, 'terminal_field': (terminal @ w).reshape((study['nx'], study['ny']), order='F'),
            'kkt_certificate': certificate}
