"""Reaction-transport source inversion with a box-simplex optimality certificate."""

from pathlib import Path
import json

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _archived_affine(bank):
    history = json.loads(_CALIBRATION_ARCHIVE.read_text())
    ref, raw = [], []
    for sweep in history:
        mask = np.asarray(sweep['bank_id']) == bank
        ref.extend(np.asarray(sweep['ref_signal'], dtype=float)[mask])
        raw.extend(np.asarray(sweep['raw_readout'], dtype=float)[mask])
    return _affine(ref, raw)


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    y = np.empty(len(row_banks))
    for bank in np.unique(row_banks):
        x, z = ref[banks == bank], raw[banks == bank]
        if len(x) < 2:
            gain, offset = _archived_affine(bank)
        else:
            gain, offset = _affine(x, z)
            if len(x) == 4:
                # One spike contaminates half the pairwise slopes. Fit each
                # three-row subset and test the omitted row of the best fit.
                best_error = np.inf
                for omitted in range(4):
                    kept = np.arange(4) != omitted
                    trial_gain, trial_offset = _affine(x[kept], z[kept])
                    residual = z - (trial_gain * x + trial_offset)
                    error = residual[kept] @ residual[kept]
                    if error < best_error:
                        best_error = error
                        omitted_residual = residual[omitted]
                if abs(omitted_residual) > 3.0:
                    gain, offset = _archived_affine(bank)
            elif len(x) == 3:
                # With three points, every two-point fit can have zero error.
                # Test the omitted residual for each identifiable affine fit
                # instead of letting the median slope mask an endpoint spike.
                for omitted in range(3):
                    i, j = [k for k in range(3) if k != omitted]
                    if x[i] == x[j]:
                        continue
                    predicted = z[i] + (z[j] - z[i]) * (x[omitted] - x[i]) / (x[j] - x[i])
                    if abs(z[omitted] - predicted) > 3.0:
                        gain, offset = _archived_affine(bank)
                        break
            elif len(x) >= 5:
                i, j = np.triu_indices(len(x), 1)
                distinct = x[i] != x[j]
                i, j = i[distinct], j[distinct]
                if len(i):
                    slope = np.median((z[i] - z[j]) / (x[i] - x[j]))
                    intercept = np.median(z - slope * x)
                    clean = np.abs(z - slope * x - intercept) <= 3.0
                    if not np.all(clean):
                        if clean.sum() >= 4:
                            gain, offset = _affine(x[clean], z[clean])
                        else:
                            gain, offset = _archived_affine(bank)
        take = row_banks == bank
        y[take] = (np.asarray(study['y_obs_raw'])[take] - offset) / gain
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


def _periodic_axis_operator(length, kappa, velocity):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    kappa = np.asarray(kappa, dtype=float)
    face = (kappa + kappa[ip]) / (2 * h**2)
    v = np.asarray(velocity, dtype=float) / h
    values = np.r_[ -face[im] - np.maximum(v, 0),
                    face[im] + face + np.abs(v),
                   -face - np.maximum(-v, 0)]
    return coo_matrix((values, (np.tile(i, 3), np.r_[im, i, ip])),
                      shape=(n, n)).tocsc()


def _forward(study):
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    trace = np.asarray(study['past_vortex_circulation_trace'], dtype=float)
    excess = np.maximum(trace - study['critical_vortex_circulation'], 0.)
    memory = float(excess @ excess)
    loop = study['recirculation_loop_mode']
    periodic = loop > 0.50
    sealed = loop == 0 and memory > 0
    potential = np.asarray(study['V_reaction_2d'], dtype=float)
    if periodic:
        x = np.linspace(0, 2 * np.pi, nx)[:, None]
        y = np.linspace(0, 2 * np.pi, ny)[None, :]
        potential = potential + 0.75 * loop * np.cos(x) * np.cos(y)
    elif sealed:
        lx, ly = study['Lx'], study['Ly']
        x = np.linspace(0, lx, nx)[:, None]
        y = np.linspace(0, ly, ny)[None, :]
        wall_mask = np.exp(-(x / (0.25 * lx))**2) + np.exp(-((lx - x) / (0.25 * lx))**2)
        potential = potential + (0.95 + 0.55 * np.tanh(memory)) * wall_mask * (0.6 + 0.4 * np.sin(np.pi * y / ly))
    axes = [_axis_operator(study['L' + d], study['kappa_' + d],
                           study['vel_' + d], 0. if sealed else study['robin_' + d]) for d in 'xy']
    if periodic:
        axes = [_periodic_axis_operator(study['L' + d], study['kappa_' + d],
                                        study['vel_' + d]) for d in 'xy']
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx))
          + diags(potential.ravel(order='F'))).tocsc()
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


def solve_beta_m97_c01_batch_03(study: dict) -> dict:
    """Recover box-simplex source weights and the last-shift field."""
    y = _calibrated_observations(study)
    S, fields = _forward(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((y, S))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residualized = values - Z @ np.linalg.solve(gram, Z.T @ values)
    y, S = residualized[:, 0], residualized[:, 1:]
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
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
                   'primal_objective': primal, 'dual_lower_bound': float(primal - gap)}
    field = (fields @ w).reshape(study['source_modes'].shape[1:], order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': certificate}
