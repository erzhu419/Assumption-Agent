"""Separable tracer transport inversion with a box-simplex certificate."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k, v = np.asarray(kappa), np.asarray(vel)
    diffusion = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    main = np.array(diag_add, dtype=float, copy=True) + np.abs(v) / h
    main[:-1] += diffusion
    main[1:] += diffusion
    ends = [0, n - 1]
    main[ends] += robin * k[ends] / (h * (robin * h / 2 + k[ends]))
    lower = -diffusion - np.maximum(v[1:], 0) / h
    upper = -diffusion + np.minimum(v[:-1], 0) / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    s = study
    modes = np.asarray(s['source_modes'])
    m, nx, ny = modes.shape
    V = np.asarray(s['V_reaction_2d'])
    vx = V.mean(axis=1)
    vy = V.mean(axis=0) - V.mean()
    ax = _build_1d(s['Lx'], nx, s['kappa_x'], s['vel_x'], s['robin_x'],
                   vx + s['reaction_x'])
    by = _build_1d(s['Ly'], ny, s['kappa_y'], s['vel_y'], s['robin_y'],
                   vy + s['reaction_y'])
    operator = (kron(ax, eye(ny)) + kron(eye(nx), by)).tocsc()
    ident = eye(nx * ny, format='csc')
    rhs = modes.reshape(m, -1).T
    i, j = np.asarray(s['sensor_ij']).T
    blocks = []
    for mu in s['mu_shifts']:
        fields = splu(operator + mu * ident).solve(rhs).reshape(nx, ny, m)
        blocks.append(fields[i, j, :])
    return np.concatenate(blocks, axis=0), fields


def _fit_sweep(x, y):
    X = np.column_stack((x, np.ones(len(x))))
    fit = np.linalg.lstsq(X, y, rcond=None)[0]
    if len(x) < 4:
        return fit
    i, j = np.triu_indices(len(x), 1)
    keep = x[i] != x[j]
    i, j = i[keep], j[keep]
    if not len(i):
        return fit
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    intercepts = y[i] - slopes * x[i]
    residuals = np.abs(y[None, :] - slopes[:, None] * x - intercepts[:, None])
    scale = np.median(residuals, axis=1)
    best = scale.argmin()
    resid = residuals[best]
    inlier = resid <= max(5.0 * scale[best], 1e-7 * max(1.0, np.max(np.abs(y[resid <= np.median(resid)]))))
    if np.ptp(x[inlier]) > 0:
        return np.linalg.lstsq(X[inlier], y[inlier], rcond=None)[0]
    return fit


def _calibrate(study):
    c = study['sensor_calibration_sweeps']
    bank, ref, raw = (np.asarray(c[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    gains, offsets = np.ones(4), np.zeros(4)
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        if len(x) and np.ptp(x) > 0:
            gains[b], offsets[b] = _fit_sweep(x, y)
        elif len(x):
            offsets[b] = np.median(y - x)
    row_bank = np.asarray(study['row_bank_id'])
    return (np.asarray(study['y_obs_raw']) - offsets[row_bank]) / gains[row_bank]


def _profile(Z, ridge, data):
    gram = Z.T @ Z + ridge * np.eye(Z.shape[1])
    return data - Z @ np.linalg.solve(gram, Z.T @ data)


def _solve_box_simplex(H, c, upper, budget):
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(np.min(c))
    if budget == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w + c))
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)
    tol = 1e-12
    for _ in range(100 * n):
        free = np.flatnonzero(status == 0)
        g = H @ w + c
        if not len(free):
            lo = np.flatnonzero(status == -1)
            hi = np.flatnonzero(status == 1)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[i] >= g[j] - tol:
                return w, float((g[i] + g[j]) / 2)
            status[[i, j]] = 0
            continue
        Q = H[np.ix_(free, free)]
        ones = np.ones(len(free))
        a, b = np.linalg.solve(Q, np.column_stack((ones, g[free]))).T
        nu = (b.sum() + budget - w.sum()) / a.sum()
        direction = nu * a - b
        target = w[free] + direction
        down = (direction < 0) & (target < -tol)
        up = (direction > 0) & (target > upper + tol)
        if np.any(down | up):
            step = np.full(len(free), np.inf)
            step[down] = -w[free[down]] / direction[down]
            step[up] = (upper - w[free[up]]) / direction[up]
            k = np.argmin(step)
            w[free] += step[k] * direction
            hit = free[k]
            status[hit] = -1 if down[k] else 1
            w[hit] = 0.0 if down[k] else upper
            continue
        w[free] = np.clip(target, 0.0, upper)
        g = H @ w + c
        violation = status * (g - nu)
        i = np.argmax(violation)
        if violation[i] <= tol:
            return w, float(nu)
        status[i] = 0
    raise RuntimeError('Box-simplex active set failed to converge')


def solve_beta_w03_s00(study: dict) -> dict:
    S, terminal = _forward(study)
    y = _calibrate(study)
    S_and_y = _profile(np.asarray(study['Z_nuisance']), study['nuisance_ridge'],
                       np.column_stack((S, y)))
    S, y = S_and_y[:, :-1], S_and_y[:, -1]
    ridge, upper, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _solve_box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, upper, budget)
    res = S @ w - y
    g = S.T @ res + ridge * w
    primal = float(0.5 * (res @ res + ridge * (w @ w)))
    dual = float(primal - g @ w + nu * budget + upper * np.minimum(g - nu, 0).sum())
    return {'weights': w, 'terminal_field': terminal @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(g - nu, 0),
                                'lam_upper': np.maximum(nu - g, 0),
                                'primal_objective': primal,
                                'dual_lower_bound': min(primal, dual)}}
