"""Fouled-membrane reactor tomography and box-simplex KKT inversion."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrated_target(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'])
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    result = np.empty_like(obs)
    archive = None
    for bank in range(4):
        rows = banks == bank
        x, y = ref[rows], raw[rows]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, 1)
            dx = x[j] - x[i]
            distinct = dx != 0.0
            if np.any(distinct):
                slope = np.median((y[j] - y[i])[distinct] / dx[distinct])
                offsets = y - slope * x
                clean = np.abs(offsets - np.median(offsets)) <= 3.0
        n_clean = np.count_nonzero(clean)
        if n_clean >= 4 or (n_clean >= 2 and np.all(clean)):
            gain, offset = _fit_line(x[clean], y[clean])
        else:
            if archive is None:
                archive = json.loads(_CALIBRATION_ARCHIVE.read_text())
            for record in reversed(archive):
                mask = np.asarray(record['bank_id']) == bank
                ax = np.asarray(record['ref_signal'], dtype=float)[mask]
                if np.unique(ax).size >= 2:
                    ay = np.asarray(record['raw_readout'], dtype=float)[mask]
                    gain, offset = _fit_line(ax, ay)
                    break
            else:
                raise ValueError(f'Insufficient calibration data for bank {bank}')
        rows = row_banks == bank
        result[rows] = (obs[rows] - offset) / gain
    return result


def _axis_operator(kappa, velocity, length, robin):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    v = velocity[1:-1]
    lower, upper = -faces.copy(), -faces.copy()
    lower[:-1] -= np.maximum(v, 0) / h
    upper[1:] -= np.maximum(-v, 0) / h
    upper[0], lower[-1] = -kappa[[0, -1]] / h**2
    main = np.empty(n)
    main[1:-1] = faces[:-1] + faces[1:] + np.abs(v) / h
    main[[0, -1]] = kappa[[0, -1]] / h**2 + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    lx, ly = study['Lx'], study['Ly']
    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'], dtype=float)
                        - study['critical_vortex_circulation'])
    memory = float(np.sum(excess**2))
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0.0
    ax = _axis_operator(study['kappa_x'], study['vel_x'], lx,
                        0.0 if sealed else study['robin_x'])
    ay = _axis_operator(study['kappa_y'], study['vel_y'], ly,
                        0.0 if sealed else study['robin_y'])
    potential = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        x = np.linspace(0.0, lx, nx)[:, None]
        y = np.linspace(0.0, ly, ny)[None, :]
        walls = np.exp(-(x / (0.25 * lx))**2) + np.exp(-((lx - x) / (0.25 * lx))**2)
        potential = potential + (0.95 + 0.55 * np.tanh(memory)) * walls * (0.6 + 0.4 * np.sin(np.pi * y / ly))
    potential = potential.ravel(order='F')
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(potential, format='csc'))
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.transpose(0, 2, 1).reshape(modes.shape[0], -1).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_index = i + nx * j
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(operator + float(mu) * identity).solve(rhs)
        sampled.append(field[sensor_index])
    return np.vstack(sampled), field


def _box_simplex(design, target, ridge, upper, budget):
    """Feasible active-set QP, releasing bounds with the wrong KKT sign."""
    n = design.shape[1]
    hessian = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    tol = 1e-12
    if budget == 0:
        w = np.zeros(n)
        return w, float(np.min(-linear))
    if budget == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(hessian @ w - linear))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1: lower, +1: upper
    for _ in range(500):
        free = np.flatnonzero(active == 0)
        lo, hi = active < 0, active > 0
        grad = hessian @ w - linear
        if free.size == 0:
            i = np.argmin(np.where(lo, grad, np.inf))
            j = np.argmax(np.where(hi, grad, -np.inf))
            nu = (grad[i] + grad[j]) / 2
            if grad[i] >= grad[j] - tol:
                return w, float(nu)
            active[[i, j]] = 0
            continue

        # Minimize on the current face with its equality constraint intact.
        system = np.ones((free.size + 1, free.size + 1))
        system[:-1, :-1] = hessian[np.ix_(free, free)]
        system[-1, -1] = 0.0
        rhs = np.r_[linear[free] - hessian[free][:, hi] @ w[hi], budget - w[hi].sum()]
        face = np.linalg.solve(system, rhs)
        candidate = face[:-1]
        direction = candidate - w[free]
        ratios = np.full(free.size, np.inf)
        below, above = candidate < -tol, candidate > upper + tol
        ratios[below] = -w[free[below]] / direction[below]
        ratios[above] = (upper - w[free[above]]) / direction[above]
        k = int(np.argmin(ratios))
        if ratios[k] < 1:
            w[free] += max(0.0, ratios[k]) * direction
            active[free[k]] = -1 if below[k] else 1
            w[free[k]] = 0.0 if below[k] else upper
            continue

        w[free] = np.clip(candidate, 0.0, upper)
        nu = -face[-1]
        grad = hessian @ w - linear
        violation = np.where(lo, nu - grad, np.where(hi, grad - nu, 0.0))
        k = int(np.argmax(violation))
        if violation[k] <= tol:
            return w, float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_batch_01(study: dict) -> dict:
    target = _calibrated_target(study)
    sensor, field = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    stacked = np.column_stack((target, sensor))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = stacked - z @ np.linalg.solve(gram, z.T @ stacked)
    target, sensor = projected[:, 0], projected[:, 1:]
    ridge, upper, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(sensor, target, ridge, upper, budget)
    residual = sensor @ w - target
    grad = sensor.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = grad - nu
    lower_bound = objective - grad @ w + nu * budget + upper * np.minimum(reduced, 0.0).sum()
    cert = {
        'weights': w.tolist(), 'nu_sum': nu,
        'lam_lower': np.maximum(reduced, 0.0).tolist(),
        'lam_upper': np.maximum(-reduced, 0.0).tolist(),
        'primal_objective': float(objective), 'dual_lower_bound': float(lower_bound),
    }
    terminal = (field @ w).reshape((study['nx'], study['ny']), order='F')
    return {'weights': w, 'terminal_field': terminal, 'kkt_certificate': cert}
