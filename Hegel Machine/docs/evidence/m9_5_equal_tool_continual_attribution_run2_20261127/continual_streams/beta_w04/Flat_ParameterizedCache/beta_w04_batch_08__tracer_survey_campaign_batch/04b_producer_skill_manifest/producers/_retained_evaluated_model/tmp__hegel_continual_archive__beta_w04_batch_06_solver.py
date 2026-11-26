"""Nominal tracer transport and calibrated, nuisance-profiled inversion."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _bank_sweeps(sweeps):
    bank, x, y = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    return [(x[bank == b], y[bank == b]) for b in range(4)]


def _fit_bank(x, y):
    """Fit an affine line, rejecting isolated EMI spikes on redundant sweeps."""
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    distinct = np.abs(dx) > 1e-12
    if not distinct.any():
        return None
    if x.size >= 4:
        slopes = (y[j[distinct]] - y[i[distinct]]) / dx[distinct]
        offsets = y - slopes[:, None] * x
        err = np.abs(offsets - np.median(offsets, axis=1)[:, None])
        score = np.median(err, axis=1)
        if np.any(slopes > 0):
            best = np.argmin(np.where(slopes > 0, score, np.inf))
            keep = err[best] <= max(1e-6, 6 * score[best])
            if np.ptp(x[keep]) > 1e-12:
                x, y = x[keep], y[keep]
    xc = x - x.mean()
    gain = float(xc @ (y - y.mean()) / (xc @ xc))
    return np.array([gain, y.mean() - gain * x.mean()])


def _calibration(study):
    current = _bank_sweeps(study['sensor_calibration_sweeps'])
    fits = [_fit_bank(x, y) for x, y in current]
    if all(p is not None and p[0] > 0 and len(x) >= 4 for p, (x, _) in zip(fits, current)):
        return np.stack(fits)
    # Use complete historical sweeps as joint profiles: a spot check can select
    # an instrument state but cannot identify gains on the unchecked banks.
    with _HISTORY.open() as f:
        sweeps = [_bank_sweeps(e) for e in json.load(f)]
    profiles = []
    for sweep in sweeps:
        p = [_fit_bank(x, y) for x, y in sweep]
        if all(q is not None and q[0] > 0 for q in p):
            profiles.append(np.stack(p))

    def agreement(p):
        residuals = [np.abs(y - (a * x + c)) / np.maximum(1., np.abs(y))
                     for (a, c), (x, y) in zip(p, current) if len(x)]
        return (sum(np.mean(r <= 1e-5) for r in residuals),
                -sum(np.minimum(np.median(r), 1e-5) for r in residuals))

    anchor = max(reversed(profiles), key=agreement)
    similar = [p for p in profiles if np.max(np.abs(p - anchor)) <= 1e-5]
    prior = np.median(similar, axis=0)
    for b, ((x, y), fit) in enumerate(zip(current, fits)):
        if fit is None or fit[0] <= 0:
            continue
        a, c = prior[b]
        n_agree = np.count_nonzero(np.abs(y - (a*x + c)) <= 1e-5*np.maximum(1., np.abs(y)))
        if len(x) >= 4 or n_agree != 1 or len(x) != 2:
            prior[b] = fit
    return prior


def _calibrate(study):
    gains, offsets = _calibration(study).T
    b = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[b]) / gains[b]


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    p = np.maximum(vel, 0) / h
    m = np.maximum(-np.asarray(vel), 0) / h
    q = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    d = p + m + diag_add
    d[:-1] += q
    d[1:] += q
    edge = np.array([0, n - 1])
    np.add.at(d, edge, robin * k[edge] / (h * (robin * h / 2 + k[edge])))
    return diags((-q - p[1:], d, -q - m[:-1]), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx, vy = v.mean(axis=1), v.mean(axis=0) - v.mean()
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'],
                   study['robin_x'], study['reaction_x'] + vx)
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'],
                   study['robin_y'], study['reaction_y'] + vy)
    # Row-major vectorization of Ax @ U + U @ By.T.
    op = (kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc'))
    ident = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.reshape(len(modes), -1).T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    blocks = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * ident).solve(rhs).reshape(nx, ny, len(modes))
        blocks.append(field[i, j])
    return np.concatenate(blocks), field


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


def solve_beta_w04_batch_06(study: dict) -> dict:
    design, terminal = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    profiled = _profile(z, np.column_stack((design, y)), float(study['nuisance_ridge']))
    s, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    h = s.T @ s + ridge * np.eye(s.shape[1])
    w, nu = _box_simplex(h, -(s.T @ r), cap, budget)
    res = s @ w - r
    g = s.T @ res + ridge * w
    primal = 0.5 * float(res @ res + ridge * (w @ w))
    lower = primal - g @ w + nu * budget + cap * np.minimum(g - nu, 0.).sum()
    return {'weights': w, 'terminal_field': terminal @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(g - nu, 0.),
                                'lam_upper': np.maximum(nu - g, 0.),
                                'primal_objective': primal, 'dual_lower_bound': float(lower)}}
