"""Calibrated tracer inversion with an explicit box-simplex KKT certificate."""
import json
from pathlib import Path

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


_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _bank_history(bank):
    try:
        with _CALIBRATION_HISTORY.open() as stream:
            history = json.load(stream)
    except FileNotFoundError:
        return 1.0, 0.0
    # Use the most recent sweep that identifies both calibration parameters.
    for sweep in reversed(history):
        keep = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'], dtype=float)[keep]
        y = np.asarray(sweep['raw_readout'], dtype=float)[keep]
        if x.size >= 2 and np.ptp(x) > 1e-12:
            return _fit_bank(x, y)
    return 1.0, 0.0


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gains, offsets = np.empty(4), np.empty(4)
    for b in range(4):
        keep = banks == b
        x, y = ref[keep], raw[keep]
        if x.size >= 2 and np.ptp(x) > 1e-12:
            if x.size == 3:
                # With one bad spot, the three candidate pairs each define a line.
                # Keep the current gain if all three agree; otherwise use the
                # historical gain and let the two clean spots anchor the offset.
                xc, yc = x - x.mean(), y - y.mean()
                gain = (xc @ yc) / (xc @ xc)
                if np.max(np.abs(yc - gain * xc)) > 1e-6 * max(1.0, np.max(np.abs(y))):
                    gain, _ = _bank_history(b)
                    gains[b], offsets[b] = gain, np.median(y - gain * x)
                    continue
            gains[b], offsets[b] = _fit_bank(x, y)
        else:
            gain, offset = _bank_history(b)
            gains[b] = gain
            offsets[b] = np.median(y - gain * x) if x.size else offset
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[rows]) / gains[rows]


def _forward(study):
    nx, ny = study['nx'], study['ny']
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'],
                   v.mean(axis=1) + np.asarray(study['reaction_x']))
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'],
                   v.mean(axis=0) - v.mean() + np.asarray(study['reaction_y']))
    op = (kron(ax, eye(ny)) + kron(eye(nx), by)).tocsc()
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.reshape(len(modes), -1).T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * identity).solve(rhs).reshape(nx, ny, len(modes))
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


def solve_beta_w04_batch_01(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    data = _profile(z, np.column_stack((design, y)), float(study['nuisance_ridge']))
    S, r = data[:, :-1], data[:, -1]
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    c = -(S.T @ r)
    weights, certificates = [], []
    for total in study['budget_grid']:
        budget = float(total)
        w, nu = _box_simplex(H, c, cap, budget)
        residual = S @ w - r
        g = S.T @ residual + ridge * w
        primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
        lower = np.maximum(g - nu, 0.0)
        upper = np.maximum(nu - g, 0.0)
        bound = primal + nu * (budget - w.sum()) - lower @ w - upper @ (cap - w)
        weights.append(w)
        certificates.append({'weights': w.copy(), 'nu_sum': nu,
                             'lam_lower': lower, 'lam_upper': upper,
                             'primal_objective': primal, 'dual_lower_bound': float(bound)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': terminal_modes @ weights[1],
            'kkt_certificates': certificates}
