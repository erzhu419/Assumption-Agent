"""Calibrated tracer inversion with an explicit box-simplex KKT certificate."""
import json

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    p = np.maximum(vel, 0) / h
    m = np.maximum(-np.asarray(vel), 0) / h
    q = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    diagonal = p + m + diag_add
    diagonal[:-1] += q
    diagonal[1:] += q
    edge = np.array([0, n - 1])
    np.add.at(diagonal, edge, robin * k[edge] / (h * (robin * h / 2 + k[edge])))
    return diags((-q - p[1:], diagonal, -q - m[:-1]), (-1, 0, 1), format='csc')


def _fit_bank(x, y):
    if not x.size:
        return 1.0, 0.0
    a, b = np.triu_indices(x.size, 1)
    dx = x[b] - x[a]
    distinct = np.abs(dx) > 1e-12 * max(1.0, np.max(np.abs(x)))
    if not distinct.any():
        return 1.0, float(np.median(y - x))
    keep = np.ones(x.size, dtype=bool)
    if x.size >= 4:
        # A median-residual line prevents a single EMI spike from setting the gain.
        slopes = (y[b[distinct]] - y[a[distinct]]) / dx[distinct]
        slopes = slopes[slopes > 0]
        if slopes.size:
            diffs = y - slopes[:, None] * x
            centers = np.median(diffs, axis=1)
            err = np.abs(diffs - centers[:, None])
            best = np.argmin(np.median(err, axis=1))
            scale = max(1.4826 * np.median(err[best]), 1e-12 * max(1.0, np.max(np.abs(y))))
            keep = err[best] <= 4.5 * scale
            if np.ptp(x[keep]) <= 1e-12:
                keep[:] = True
    x, y = x[keep], y[keep]
    xc = x - x.mean()
    gain = float(xc @ (y - y.mean()) / (xc @ xc))
    if gain <= 0:
        return 1.0, float(np.median(y - x))
    return gain, float(y.mean() - gain * x.mean())


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    fits = []
    archived = None
    for bank in range(4):
        x, y = ref[banks == bank], raw[banks == bank]
        if x.size and np.ptp(x) > 1e-12 * max(1.0, np.max(np.abs(x))):
            fits.append(_fit_bank(x, y))
            continue
        if archived is None:
            # Batch 02 used the same calibration as this campaign; the intervening
            # batch used unit gains. Only borrow when the current gain is unidentified.
            with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
                archived = next(e for e in json.load(f)
                                if e['episode_short_name'] == 'beta_w04_batch_02')
        mask = np.asarray(archived['bank_id']) == bank
        historical_gain, historical_offset = _fit_bank(
            np.asarray(archived['ref_signal'], dtype=float)[mask],
            np.asarray(archived['raw_readout'], dtype=float)[mask],
        )
        fits.append((historical_gain, float(np.median(y - historical_gain * x))
                     if x.size else historical_offset))
    gain, offset = np.asarray(fits).T
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset[rows]) / gain[rows]


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _build_1d(study['Lx'], nx, 1.45 * np.asarray(study['kappa_x']),
                   study['vel_x'], study['robin_x'], np.asarray(study['reaction_x']))
    by = _build_1d(study['Ly'], ny, 1.45 * np.asarray(study['kappa_y']),
                   study['vel_y'], study['robin_y'], np.asarray(study['reaction_y']))
    potential = np.asarray(study['V_reaction_2d'], dtype=float).ravel()
    op = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.reshape(len(modes), -1).T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = []
    for mu in study['mu_shifts']:
        shifted = op + diags(potential + mu, format='csc')
        field = splu(shifted).solve(rhs).reshape(nx, ny, len(modes))
        sensor_rows.append(field[i, j])
    return np.concatenate(sensor_rows), field


def _profile(z, data, ridge):
    if ridge == 0:
        coef = np.linalg.lstsq(z, data, rcond=None)[0]
    else:
        coef = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ data)
    return data - z @ coef


def _box_simplex(H, c, cap, budget):
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(np.min(c))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w + c))
    w = np.full(n, budget / n)
    # status -1/0/1 means fixed at lower/free/fixed at upper.
    status = np.zeros(n, dtype=int)
    for _ in range(100 * n):
        g = H @ w + c
        free = np.flatnonzero(status == 0)
        if not free.size:
            lo = np.where(status < 0, g, np.inf).argmin()
            hi = np.where(status > 0, g, -np.inf).argmax()
            if g[lo] >= g[hi] - 1e-12:
                return w, float((g[lo] + g[hi]) / 2)
            status[[lo, hi]] = 0
            continue
        # Newton step to the minimizer on the current sum-preserving face.
        block = H[np.ix_(free, free)]
        one = np.ones(len(free))
        dg, de = np.linalg.solve(block, np.column_stack((g[free], one))).T
        nu = dg.sum() / de.sum()
        step = nu * de - dg
        alpha = np.full(len(free), np.inf)
        np.divide(cap - w[free], step, out=alpha, where=step > 1e-14)
        np.divide(-w[free], step, out=alpha, where=step < -1e-14)
        nearest = int(alpha.argmin())
        if alpha[nearest] < 1.0:
            w[free] += max(0.0, alpha[nearest]) * step
            hit = free[nearest]
            status[hit] = 1 if step[nearest] > 0 else -1
            w[hit] = cap if status[hit] > 0 else 0.0
            continue
        w[free] += step
        g = H @ w + c
        violation = status * (g - nu)
        release = int(violation.argmax())
        if violation[release] <= 1e-12:
            return np.clip(w, 0.0, cap), float(nu)
        status[release] = 0
    raise RuntimeError('Box-simplex active-set solver did not converge')


def solve_beta_w04_batch_04(study: dict) -> dict:
    raw_design, terminal_modes = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    profiled = _profile(z, np.column_stack((raw_design, y)), float(study['nuisance_ridge']))
    S, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S.T @ r), cap, budget)
    residual = S @ w - r
    g = S.T @ residual + ridge * w
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = primal - g @ w + nu * budget + cap * np.minimum(g - nu, 0.0).sum()
    return {'weights': w, 'terminal_field': terminal_modes @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(g - nu, 0.0),
                                'lam_upper': np.maximum(nu - g, 0.0),
                                'primal_objective': primal, 'dual_lower_bound': float(lower)}}
