"""Calibrated multi-shift tracer tomography with a box-simplex certificate."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    face = (2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:])) / h**2
    positive, negative = np.maximum(vel, 0) / h, np.maximum(-vel, 0) / h
    center = np.r_[face, 0.] + np.r_[0., face] + diag_add + positive + negative
    edge = kappa[[0, -1]]
    center[[0, -1]] += robin * edge / (h * (0.5 * robin * h + edge))
    return diags((-face - positive[1:], center, -face - negative[:-1]),
                 (-1, 0, 1), shape=(n, n), format='csc')


import json
from pathlib import Path


def _responses(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'],
                   study['reaction_x'] + v.mean(axis=1))
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'],
                   study['reaction_y'] + v.mean(axis=0) - v.mean())
    operator = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    ident = eye(nx * ny, format='csc')
    rhs = sources.reshape(m, -1).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    sensor = ix * ny + iy
    rows = []
    for shift in study['mu_shifts']:
        fields = splu(operator + float(shift) * ident).solve(rhs)
        rows.append(fields[sensor])
    return np.vstack(rows), fields.reshape(nx, ny, m)


def _profile(z, values, ridge):
    if z.shape[1] == 0:
        return values
    q, singular, _ = np.linalg.svd(z, full_matrices=False)
    if ridge == 0:
        q = q[:, singular > np.finfo(float).eps * max(z.shape) * singular[0]]
        return values - q @ (q.T @ values)
    shrink = singular**2 / (singular**2 + ridge)
    return values - q @ (shrink[:, None] * (q.T @ values))


def _box_simplex(hessian, linear, cap, budget):
    """Feasible active-set method; nu follows the certificate's sign convention."""
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        gradient = hessian @ w + linear
        return w, float(gradient.min() if budget == 0 else gradient.max())
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    eps = 1e-12
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        if not len(free):
            g = hessian @ w + linear
            low, high = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = low[np.argmin(g[low])], high[np.argmax(g[high])]
            if g[j] <= g[i] + eps:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            free = np.array([i, j])
        fixed = np.flatnonzero(active)
        count = len(free)
        kkt = np.ones((count + 1, count + 1))
        kkt[:count, :count] = hessian[np.ix_(free, free)]
        kkt[-1, -1] = 0.
        rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        solution = np.linalg.solve(kkt, rhs)
        goal, nu = solution[:-1], float(-solution[-1])
        direction = goal - w[free]
        crossing = np.ones(count)
        lower, upper = goal < -eps, goal > cap + eps
        crossing[lower] = -w[free[lower]] / direction[lower]
        crossing[upper] = (cap - w[free[upper]]) / direction[upper]
        hit = int(np.argmin(crossing))
        if crossing[hit] < 1.:
            w[free] += max(0., crossing[hit]) * direction
            index = free[hit]
            active[index] = -1 if lower[hit] else 1
            w[index] = 0. if lower[hit] else cap
            continue
        w[free] = np.clip(goal, 0., cap)
        violation = active * (hessian @ w + linear - nu)
        index = int(np.argmax(violation))
        if violation[index] <= eps:
            return w, nu
        active[index] = 0
    raise RuntimeError('Box-simplex working set did not converge')


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_calibration(x, y):
    """Fit the largest consensus of the affine reference sweep, then refit it."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    distinct = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x), initial=0.))
    i, j, dx = i[distinct], j[distinct], dx[distinct]
    slopes = (y[j] - y[i]) / dx
    i, slopes = i[slopes > 0], slopes[slopes > 0]
    if not slopes.size:
        return None
    deviations = np.abs(y[None, :] - y[i, None] - slopes[:, None] * (x[None, :] - x[i, None]))
    # Archived sweeps are rounded to six decimals; current sweeps can be finer.
    inliers = deviations <= 2e-6 * np.maximum(1., slopes[:, None])
    counts = inliers.sum(axis=1)
    winners = np.flatnonzero(counts == counts.max())
    best = winners[np.argmin(np.where(inliers[winners], deviations[winners], 0.).sum(axis=1))]
    keep = inliers[best]
    if len(x) > 2 and keep.sum() <= len(x) // 2:
        return None
    xc = x[keep] - x[keep].mean()
    gain = (xc @ (y[keep] - y[keep].mean())) / (xc @ xc)
    return float(gain), float(y[keep].mean() - gain * x[keep].mean())


def _historical_calibration(history, bank):
    for epoch in reversed(history):
        select = np.asarray(epoch['bank_id']) == bank
        fit = _fit_calibration(np.asarray(epoch['ref_signal'])[select],
                               np.asarray(epoch['raw_readout'])[select])
        if fit is not None:
            return fit
    return 1., 0.


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, ref, raw = (np.asarray(sweeps[key]) for key in ('bank_id', 'ref_signal', 'raw_readout'))
    rows = np.asarray(study['row_bank_id'], dtype=int)
    observed = np.asarray(study['y_obs_raw'], dtype=float)
    result = np.empty_like(observed)
    history = None
    for bank in np.unique(rows):
        current = (banks == bank) & np.isfinite(ref) & np.isfinite(raw)
        x, y = ref[current], raw[current]
        fit = _fit_calibration(x, y)
        if fit is None or x.size == 2:
            if history is None:
                history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
            gain, offset = _historical_calibration(history, bank)
            residual = y - gain * x
            # Compare in readout units: dividing by a short reference span
            # magnifies noise and archive rounding into an apparent gain change.
            if fit is not None and np.ptp(residual) > 2e-7 * max(1., gain):
                gain, offset = fit
            elif x.size:
                offset = np.median(residual)
        else:
            gain, offset = fit
        take = rows == bank
        result[take] = (observed[take] - offset) / gain
    return result


def solve_beta_w01_batch_01(study: dict) -> dict:
    design, terminal = _responses(study)
    y = _calibrated_observations(study)
    profile = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)),
                       float(study['nuisance_ridge']))
    s, r = profile[:, :-1], profile[:, -1]
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = s.T @ s + ridge * np.eye(s.shape[1])
    linear = -(s.T @ r)
    weights, certs = [], []
    for budget in study['budget_grid']:
        budget = float(budget)
        w, nu = _box_simplex(hessian, linear, cap, budget)
        residual = s @ w - r
        g = s.T @ residual + ridge * w
        lower, upper = np.maximum(g - nu, 0.), np.maximum(nu - g, 0.)
        primal = 0.5 * (residual @ residual + ridge * (w @ w))
        dual = primal - g @ w + nu * budget - cap * upper.sum()
        weights.append(w)
        certs.append({'weights': w.tolist(), 'nu_sum': nu,
                      'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                      'primal_objective': float(primal), 'dual_lower_bound': float(dual)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': terminal @ weights[1],
            'kkt_certificates': certs}
