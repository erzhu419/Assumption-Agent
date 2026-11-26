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


def _calibrate(study):
    def split(sweeps):
        bank, x, y = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        return [(x[bank == b], y[bank == b]) for b in range(4)]

    def has_span(x):
        return x.size >= 2 and np.ptp(x) > 1e-12 * max(1., np.max(np.abs(x)))

    checks = split(study['sensor_calibration_sweeps'])
    priors = [(1., 0.)] * 4
    if any(not has_span(x) for x, _ in checks):
        path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        if path.is_file():
            for episode in json.loads(path.read_text()):
                for b, (x, y) in enumerate(split(episode)):
                    if has_span(x):
                        priors[b] = _fit_bank(x, y)
    params = []
    for (x, y), (gain, offset) in zip(checks, priors):
        if has_span(x):
            gain, offset = _fit_bank(x, y)
        elif x.size:
            offset = float(np.median(y - gain * x))
        params.append((gain, offset))
    gain, offset = np.asarray(params).T
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset[rows]) / gain[rows]


def _forward(study):
    nx, ny = study['nx'], study['ny']
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'],
                   np.asarray(study['reaction_x']) + vx)
    ay = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'],
                   np.asarray(study['reaction_y']) + vy)
    op = kron(ax, eye(ny), format='csc') + kron(eye(nx), ay, format='csc')
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
    profiled = _profile(z, np.column_stack((design, y)), float(study['nuisance_ridge']))
    S, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    c = -(S.T @ r)
    certs = []
    for budget in study['budget_grid']:
        w, nu = _box_simplex(H, c, cap, float(budget))
        residual = S @ w - r
        g = S.T @ residual + ridge * w
        primal = 0.5 * (residual @ residual + ridge * (w @ w))
        delta = g - nu
        lower = primal - g @ w + nu * budget + cap * np.minimum(delta, 0.).sum()
        certs.append({'weights': w, 'nu_sum': nu, 'lam_lower': np.maximum(delta, 0.),
                      'lam_upper': np.maximum(-delta, 0.), 'primal_objective': float(primal),
                      'dual_lower_bound': float(lower)})
    weights = np.stack([cert['weights'] for cert in certs])
    return {'weights_by_budget': weights, 'terminal_field_mid_budget': terminal_modes @ weights[1],
            'kkt_certificates': certs}
