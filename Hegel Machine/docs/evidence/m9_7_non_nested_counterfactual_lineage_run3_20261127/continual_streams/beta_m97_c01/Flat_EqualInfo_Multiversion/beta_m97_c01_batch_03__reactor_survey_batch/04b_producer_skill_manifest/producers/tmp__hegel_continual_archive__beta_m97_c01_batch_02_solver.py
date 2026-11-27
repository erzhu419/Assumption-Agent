"""Calibrated reactor inversion with a box-simplex optimality certificate."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _bank_sweep(sweep, bank):
    mask = np.asarray(sweep['bank_id']) == bank
    return (np.asarray(sweep['ref_signal'], dtype=float)[mask],
            np.asarray(sweep['raw_readout'], dtype=float)[mask])


def _clean_rows(x, y):
    # Clean sweeps are affine to measurement rounding. Find their consensus
    # before applying the EMI threshold so a high-leverage spike cannot hide.
    if len(x) < 3:
        return np.ones(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[i] - slopes * x[i]
    errors = np.abs(y - slopes[:, None] * x - offsets[:, None])
    counts = (errors <= 1e-5).sum(axis=1)
    best = np.argmax(np.where(counts == counts.max(), np.abs(x[j] - x[i]), -1.))
    return errors[best] <= 3.


def _archived_affine(bank):
    history = json.loads(_CALIBRATION_ARCHIVE.read_text())
    for sweep in reversed(history):
        x, y = _bank_sweep(sweep, bank)
        clean = _clean_rows(x, y)
        if clean.sum() >= 2:
            return _affine(x[clean], y[clean])
    raise ValueError(f'No archived calibration for bank {bank}')


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(study['row_bank_id'])
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    y = np.empty_like(raw)
    for bank in np.unique(banks):
        x, z = _bank_sweep(sweeps, bank)
        clean = _clean_rows(x, z)
        n = clean.sum()
        if n >= 4 or (n >= 2 and n == len(x)):
            gain, offset = _affine(x[clean], z[clean])
        else:
            gain, offset = _archived_affine(bank)
        rows = banks == bank
        y[rows] = (raw[rows] - offset) / gain
    return y


def _axis_operator(length, kappa, velocity, robin):
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    h = length / (n - 1)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    v = np.asarray(velocity, dtype=float) / h
    lower = -face - np.maximum(v[1:], 0)
    upper = -face - np.maximum(-v[:-1], 0)
    main = np.empty(n)
    main[1:-1] = face[:-1] + face[1:] + np.abs(v[1:-1])
    main[[0, -1]] = kappa[[0, -1]] / h**2 + robin / h
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    excursion = np.maximum(0., np.asarray(study['past_vortex_circulation_trace'])
                           - study['critical_vortex_circulation'])
    memory = float(excursion @ excursion)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        lx, ly = study['Lx'], study['Ly']
        xs = np.linspace(0., lx, nx)[:, None]
        ys = np.linspace(0., ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(memory)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    axes = [_axis_operator(study['L' + d], study['kappa_' + d],
                           study['vel_' + d], 0. if sealed else study['robin_' + d]) for d in 'xy']
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx))
          + diags(reaction.ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = modes.transpose(0, 2, 1).reshape(m, nx * ny).T
    sensor_i, sensor_j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = sensor_i + nx * sensor_j
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(op + mu * identity).solve(rhs)
        blocks.append(fields[sensor_rows])
    return np.vstack(blocks), fields


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        w.fill(0 if budget == 0 else cap)
        g = H @ w - c
        return w, float(np.min(g) if budget == 0 else np.max(g))
    # Active state: -1 is lower, +1 is upper, 0 is free.
    state = np.zeros(n, dtype=int)
    for _ in range(20 * n**2):
        f = np.flatnonzero(state == 0)
        a = np.flatnonzero(state != 0)
        if not len(f):
            g = H @ w - c
            low, high = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = low[np.argmin(g[low])], high[np.argmax(g[high])]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            state[[i, j]] = 0
            continue
        k = len(f)
        K = np.empty((k + 1, k + 1))
        K[:k, :k] = H[np.ix_(f, f)]
        K[:k, k], K[k, :k], K[k, k] = -1, 1, 0
        sol = np.linalg.solve(K, np.r_[c[f] - H[np.ix_(f, a)] @ w[a],
                                         budget - w[a].sum()])
        target, nu = sol[:-1], sol[-1]
        delta = target - w[f]
        below, above = target < -1e-13, target > cap + 1e-13
        if np.any(below | above):
            step = np.full(k, np.inf)
            step[below] = -w[f[below]] / delta[below]
            step[above] = (cap - w[f[above]]) / delta[above]
            hit = np.argmin(step)
            w[f] = np.clip(w[f] + step[hit] * delta, 0, cap)
            state[f[hit]] = -1 if below[hit] else 1
            w[f[hit]] = 0 if below[hit] else cap
            continue
        w[f] = np.clip(target, 0, cap)
        violation = state * (H @ w - c - nu)
        release = np.argmax(violation)
        if violation[release] <= 1e-12:
            return w, float(nu)
        state[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_batch_01(study: dict) -> dict:
    """Recover calibrated source weights and the last-shift field."""
    y = _calibrated_observations(study)
    S, fields = _forward(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((y, S))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residualized = values - Z @ np.linalg.solve(gram, Z.T @ values)
    y, S = residualized[:, 0], residualized[:, 1:]
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
    if budget == 0. or budget == S.shape[1] * cap:
        w = np.full(S.shape[1], 0. if budget == 0. else cap)
        grad = S.T @ (S @ w - y) + ridge * w
        nu = float(grad.min() if budget == 0. else grad.max())
    else:
        w, nu = _box_simplex(S, y, ridge, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    primal = float(0.5 * (r @ r + ridge * (w @ w)))
    shifted_g = g - nu
    # Evaluate the tangent bound in gap form to avoid large cancellations.
    gap = np.maximum(shifted_g, 0) @ w + np.maximum(-shifted_g, 0) @ (cap - w)
    certificate = {'weights': w.copy(), 'nu_sum': nu,
                   'lam_lower': np.maximum(shifted_g, 0),
                   'lam_upper': np.maximum(-shifted_g, 0),
                   'primal_objective': primal, 'dual_lower_bound': float(primal - gap + nu * (budget - w.sum()))}
    field = (fields @ w).reshape(study['source_modes'].shape[1:], order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': certificate}
