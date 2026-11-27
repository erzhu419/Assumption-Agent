"""Calibrated periodic and history-dependent reactor tomography with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(bank):
    with _HISTORY.open() as handle:
        records = json.load(handle)
    for sweep in reversed(records):
        banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        selected = banks == bank
        if np.count_nonzero(selected) >= 2 and np.ptp(x[selected]) > 0:
            return _fit_line(x[selected], y[selected])
    raise ValueError(f'No archived calibration for bank {bank}')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    banks, x_all, y_all = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    gains, offsets = np.empty(4), np.empty(4)
    for bank in range(4):
        x, y = x_all[banks == bank], y_all[banks == bank]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            # Repeated-median slope keeps a spike from displacing the affine fit.
            slopes = []
            for i in range(x.size):
                different = x != x[i]
                if different.any():
                    slopes.append(np.median((y[different] - y[i]) / (x[different] - x[i])))
            if slopes:
                delta = y - np.median(slopes) * x
                clean = np.abs(delta - np.median(delta)) <= 3.0
        if clean.sum() >= 4 or (x.size >= 2 and clean.all()):
            gains[bank], offsets[bank] = _fit_line(x[clean], y[clean])
        else:
            gains[bank], offsets[bank] = _historical_calibration(bank)
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[rows]) / gains[rows]


def _axis(kappa, velocity, robin, h, periodic=False):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    mat = np.zeros((n, n))
    if periodic:
        i = np.arange(n)
        im, ip = (i - 1) % n, (i + 1) % n
        km = (kappa[i] + kappa[im]) / (2.0 * h * h)
        kp = (kappa[i] + kappa[ip]) / (2.0 * h * h)
        v = velocity / h
        np.add.at(mat, (i, im), -km - np.maximum(v, 0))
        np.add.at(mat, (i, ip), -kp - np.maximum(-v, 0))
        mat[i, i] += km + kp + np.abs(v)
        return mat
    face = (kappa[1:] + kappa[:-1]) / (2.0 * h * h)
    i = np.arange(1, n - 1)
    v = velocity[i] / h
    mat[i, i - 1] = -face[:-1] - np.maximum(v, 0)
    mat[i, i + 1] = -face[1:] - np.maximum(-v, 0)
    mat[i, i] = face[:-1] + face[1:] + abs(v)
    mat[0, 0] = kappa[0] / (h * h) + robin / h
    mat[0, 1] = -kappa[0] / (h * h)
    mat[-1, -1] = kappa[-1] / (h * h) + robin / h
    mat[-1, -2] = -kappa[-1] / (h * h)
    return mat


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    lx, ly = study['Lx'], study['Ly']
    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace']) - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    mode = study['recirculation_loop_mode']
    periodic = mode > 0.50
    sealed = mode == 0 and v_mem > 0.0
    hx, hy = lx / (nx - 1), ly / (ny - 1)
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], hx, periodic)
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], hy, periodic)
    operator = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx))
    diag = np.diag_indices(nx * ny)
    reaction = np.asarray(study['V_reaction_2d'])
    if periodic:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        reaction = reaction + 0.75 * mode * np.cos(2 * np.pi * xs / lx) * np.cos(2 * np.pi * ys / ly)
    elif sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        walls = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * walls * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator[diag] += reaction.ravel(order='F')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = ix + nx * iy
    blocks = []
    for mu in study['mu_shifts']:
        shifted = operator.copy()
        shifted[diag] += mu
        fields = np.linalg.solve(shifted, rhs)
        blocks.append(fields[sensor_rows])
    return np.vstack(blocks), fields


def _box_simplex(design, target, ridge, upper, budget):
    n = design.shape[1]
    hess = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    if budget == 0:
        state[:] = -1
    elif budget == n * upper:
        state[:] = 1
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        low, high = state < 0, state > 0
        if free.size:
            # Minimize exactly on the current face with its remaining budget.
            hff = hess[np.ix_(free, free)]
            kkt = np.ones((free.size + 1, free.size + 1))
            kkt[:-1, :-1], kkt[:-1, -1], kkt[-1, -1] = hff, -1, 0
            rhs = np.r_[linear[free] - upper * hess[np.ix_(free, high)].sum(axis=1), budget - upper * high.sum()]
            solution = np.linalg.solve(kkt, rhs)
            candidate, nu = solution[:-1], solution[-1]
            direction = candidate - w[free]
            below, above = candidate < 0, candidate > upper
            if np.any(below | above):
                alpha = np.full(free.size, np.inf)
                alpha[below] = -w[free[below]] / direction[below]
                alpha[above] = (upper - w[free[above]]) / direction[above]
                hit = np.argmin(alpha)
                w[free] = np.clip(w[free] + alpha[hit] * direction, 0, upper)
                state[free[hit]] = -1 if below[hit] else 1
                w[free[hit]] = 0 if below[hit] else upper
                continue
            w[free] = candidate
        else:
            gradient = hess @ w - linear
            a = np.min(gradient[low], initial=np.inf)
            b = np.max(gradient[high], initial=-np.inf)
            if b > a + 1e-12:
                state[np.flatnonzero(low)[np.argmin(gradient[low])]] = 0
                state[np.flatnonzero(high)[np.argmax(gradient[high])]] = 0
                continue
            nu = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
        reduced = hess @ w - linear - nu
        violation = state * reduced
        if np.max(violation) <= 1e-12:
            return w, float(nu)
        state[np.argmax(violation)] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c02_batch_03(study: dict) -> dict:
    calibrated = _calibrate(study)
    forward, fields = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((calibrated, forward))
    values -= z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)
    target, design = values[:, 0], values[:, 1:]
    ridge, upper, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(design, target, ridge, upper, budget)
    residual = design @ w - target
    primal = 0.5 * float(residual @ residual + ridge * (w @ w))
    reduced = design.T @ residual + ridge * w - nu
    lower, upper_mult = np.maximum(reduced, 0), np.maximum(-reduced, 0)
    # Equivalent to the supporting-hyperplane bound, without cancellation in the gap.
    gap = float(lower @ w + upper_mult @ (upper - w) + nu * (w.sum() - budget))
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=lower, lam_upper=upper_mult,
                primal_objective=primal, dual_lower_bound=primal - max(0.0, gap))
    shape = int(study['nx']), int(study['ny'])
    return dict(weights=w, terminal_field=(fields @ w).reshape(shape, order='F'), kkt_certificate=cert)
