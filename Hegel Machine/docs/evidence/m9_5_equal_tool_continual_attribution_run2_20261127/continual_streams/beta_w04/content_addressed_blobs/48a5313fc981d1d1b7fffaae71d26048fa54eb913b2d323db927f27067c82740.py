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
    current = [_fit_bank(x, y) if has_span(x) else None for x, y in checks]
    prior = [(1., 0.)] * 4
    if any(fit is None for fit in current):
        path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        if path.is_file():
            profiles = []
            for episode in json.loads(path.read_text()):
                banks = split(episode)
                if all(has_span(x) for x, _ in banks):
                    profiles.append([_fit_bank(x, y) for x, y in banks])

            def mismatch(profile):
                # Identify the historical instrument profile from the observed banks.
                # Use spot readouts only when no current slope can be estimated.
                dense, spot = [], []
                for (x, y), fit, (g, o) in zip(checks, current, profile):
                    if fit is not None:
                        gc, oc = fit
                        dense.append(np.hypot(g - gc, o - oc))
                    elif x.size:
                        spot.append(np.median(np.abs(g * x + o - y)))
                errors = dense or spot
                return np.median(errors) if errors else 0.

            if profiles:
                prior = min(reversed(profiles), key=mismatch)
    params = []
    for (x, y), fit, (gain, offset) in zip(checks, current, prior):
        if fit is not None:
            gain, offset = fit
        elif x.size:
            offset = float(np.median(y - gain * x))
        params.append((gain, offset))
    gain, offset = np.asarray(params).T
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset[rows]) / gain[rows]


def _forward(study):
    """Solve each shifted full-grid operator for all source modes at once."""
    nx, ny = study['nx'], study['ny']
    axes = []
    for a, n in (('x', nx), ('y', ny)):
        axes.append(_build_1d(study['L' + a], n, 1.45 * np.asarray(study['kappa_' + a]),
                              study['vel_' + a], study['robin_' + a], study['reaction_' + a]))
    v = np.asarray(study['V_reaction_2d'], dtype=float).ravel(order='C')
    operator = (kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc')
                + diags(v, format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = np.asarray(study['source_modes'], dtype=float).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    design = []
    for mu in study['mu_shifts']:
        modes = splu(operator + mu * identity).solve(rhs).reshape(nx, ny, -1)
        design.append(modes[i, j])
    return np.vstack(design), modes


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
