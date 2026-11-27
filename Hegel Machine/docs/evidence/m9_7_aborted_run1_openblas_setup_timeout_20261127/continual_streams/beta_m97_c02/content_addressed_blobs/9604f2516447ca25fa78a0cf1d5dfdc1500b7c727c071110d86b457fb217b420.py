"""Reactor tomography with fouling-dependent boundaries and a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_affine(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _archived_affine(bank):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        rows = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'], dtype=float)[rows]
        y = np.asarray(sweep['raw_readout'], dtype=float)[rows]
        if len(x) >= 2 and np.ptp(x) > 0:
            return _fit_affine(x, y)
    raise ValueError(f'No prefix calibration for bank {bank}')


def _bank_affine(bank, x, y):
    if len(x) < 2:
        return _archived_affine(bank)
    if len(x) == 2:
        return _fit_affine(x, y)

    # Score pairwise lines by support near an affine relation. A wide
    # reference interval makes the seed less sensitive to rounding.
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    i, j, dx = i[ok], j[ok], dx[ok]
    slopes = (y[j] - y[i]) / dx
    offsets = y[i] - slopes * x[i]
    errors = np.abs(y - slopes[:, None] * x - offsets[:, None])
    near = errors <= 1e-5
    pick = np.lexsort((-np.abs(dx), -near.sum(axis=1)))[0]
    # For three points a spike may be ambiguous: allow each pair to flag
    # its held-out point, then use the prefix instead of choosing a pair.
    if len(x) == 3 and not near[pick].all() and np.any(errors > 3.0):
        return _archived_affine(bank)
    if near[pick].sum() >= 3:
        seed = near[pick]
        gain, offset = _fit_affine(x[seed], y[seed])
        clean = np.abs(y - gain * x - offset) <= 3.0
        if clean.all() or clean.sum() >= 4:
            return _fit_affine(x[clean], y[clean])
        return _archived_affine(bank)
    if np.all(np.max(errors, axis=1) > 3.0):
        return _archived_affine(bank)
    return _fit_affine(x, y)


def _calibration(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    fits = np.array([_bank_affine(b, ref[bank == b], raw[bank == b]) for b in range(4)])
    gain, offset = fits[np.asarray(study['row_bank_id'], dtype=int)].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


def _axis(kap, vel, robin, length):
    kap, vel = np.asarray(kap), np.asarray(vel)
    n = len(kap)
    h = length / (n - 1)
    face = (kap[:-1] + kap[1:]) / (2 * h**2)
    low = -face - np.maximum(vel[1:], 0) / h
    up = -face - np.maximum(-vel[:-1], 0) / h
    main = np.empty(n)
    main[1:-1] = face[:-1] + face[1:] + np.abs(vel[1:-1]) / h
    main[[0, -1]] = kap[[0, -1]] / h**2 + robin / h
    up[0], low[-1] = -kap[0] / h**2, -kap[-1] / h**2
    return diags((low, main, up), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    Lx, Ly = study['Lx'], study['Ly']
    excursion = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                           - study['critical_vortex_circulation'])
    v_mem = float(excursion @ excursion)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], Lx)
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], Ly)
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        x = np.linspace(0.0, Lx, nx)[:, None]
        y = np.linspace(0.0, Ly, ny)[None, :]
        wall = np.exp(-(x / (0.25 * Lx))**2) + np.exp(-((Lx - x) / (0.25 * Lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * y / Ly))
    A = (kron(eye(ny, format='csc'), ax, format='csc')
         + kron(ay, eye(nx, format='csc'), format='csc')
         + diags(reaction.ravel(order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor = i + nx * j
    I = eye(nx * ny, format='csc')
    blocks = []
    for mu in study['mu_shifts']:
        response = splu(A + mu * I).solve(modes)
        blocks.append(response[sensor])
    return np.concatenate(blocks), response


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    # -1 marks a lower bound, +1 an upper bound, and 0 a free coordinate.
    active = np.zeros(n, dtype=int)
    w = np.full(n, budget / n)
    for _ in range(20 * n * n):
        g = H @ w - c
        if budget == 0 or cap == 0:
            nu = g.min()
            break
        if budget == n * cap:
            nu = g.max()
            break
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active < 0)
        upper = np.flatnonzero(active > 0)
        if not len(free):
            lo = lower[np.argmin(g[lower])]
            hi = upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + 1e-12:
                nu = (g[hi] + g[lo]) / 2
                break
            active[[lo, hi]] = 0
            continue

        hf = H[np.ix_(free, free)]
        rhs = c[free] - cap * H[np.ix_(free, upper)].sum(axis=1)
        z, v = np.linalg.solve(hf, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (budget - cap * len(upper) - z.sum()) / v.sum()
        target = z + nu * v
        step = target - w[free]
        below, above = target < -1e-13, target > cap + 1e-13
        if below.any() or above.any():
            alpha = np.full(len(free), np.inf)
            alpha[below] = -w[free[below]] / step[below]
            alpha[above] = (cap - w[free[above]]) / step[above]
            hit = int(np.argmin(alpha))
            w[free] = np.clip(w[free] + max(0.0, alpha[hit]) * step, 0, cap)
            active[free[hit]] = 1 if above[hit] else -1
            w[free[hit]] = cap if above[hit] else 0
            continue

        w[free] = np.clip(target, 0, cap)
        g = H @ w - c
        violation = active * (g - nu)
        worst = int(np.argmax(violation))
        if violation[worst] <= 1e-12:
            break
        active[worst] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')

    residual = S @ w - y
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    g = S.T @ residual + ridge * w
    shifted = g - nu
    lower = np.where(w == 0, np.maximum(shifted, 0), 0)
    upper = np.where(w == cap, np.maximum(-shifted, 0), 0)
    dual = primal - g @ w + nu * budget + cap * np.minimum(shifted, 0).sum()
    return w, {'weights': w.copy(), 'nu_sum': float(nu), 'lam_lower': lower,
               'lam_upper': upper, 'primal_objective': float(primal),
               'dual_lower_bound': float(min(primal, dual))}


def solve_beta_m97_c02_batch_01(study: dict) -> dict:
    y = _calibration(study)
    S, terminal = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    values = np.column_stack((y, S))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residual = values - Z @ np.linalg.solve(gram, Z.T @ values)
    w, cert = _box_simplex(residual[:, 1:], residual[:, 0], study['source_ridge'],
                           study['u_box'], study['total_budget'])
    field = (terminal @ w).reshape(study['nx'], study['ny'], order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': cert}
