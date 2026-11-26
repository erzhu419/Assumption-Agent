"""Calibrated multi-wing tracer tomography with per-budget KKT certificates."""

import json
from pathlib import Path

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


def _fit_calibration(x, y):
    # Repeated-median line, followed by least squares on the inliers.
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    distinct = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x), initial=0.))
    usable = distinct.any(axis=1)
    if not usable.any():
        return None
    slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=distinct)
    gain = np.median(np.nanmedian(slopes[usable], axis=1))
    offset = np.median(y - gain * x)
    residual = y - (gain * x + offset)
    scale = 1.4826 * np.median(np.abs(residual))
    keep = np.abs(residual) <= max(4.5 * scale, 1e-10 * max(1., np.max(np.abs(y))))
    if distinct[np.ix_(keep, keep)].any():
        a, b = x[keep], y[keep]
        centered = a - a.mean()
        gain = centered @ (b - b.mean()) / (centered @ centered)
        offset = b.mean() - gain * a.mean()
    return float(gain), float(offset)


def _bank_points(sweep, bank):
    mask = np.asarray(sweep['bank_id']) == bank
    return (np.asarray(sweep['ref_signal'], dtype=float)[mask],
            np.asarray(sweep['raw_readout'], dtype=float)[mask])


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    rows = np.asarray(study['row_bank_id'], dtype=int)
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    history = None
    for bank in np.unique(rows):
        x, raw = _bank_points(sweeps, bank)
        fit = _fit_calibration(x, raw)
        if fit is None:
            if history is None:
                path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
                history = json.loads(path.read_text()) if path.is_file() else []
            gain, offset = 1., 0.
            for old_sweeps in reversed(history):
                old_fit = _fit_calibration(*_bank_points(old_sweeps, bank))
                if old_fit is not None:
                    gain, offset = old_fit
                    break
            if x.size:
                offset = float(np.median(raw - gain * x))
        else:
            gain, offset = fit
        mask = rows == bank
        y[mask] = (y[mask] - offset) / gain
    return y


def solve_beta_w01_batch_01(study: dict) -> dict:
    design, terminal = _responses(study)
    y = _calibrated_observations(study)
    values = _profile(np.asarray(study['Z_nuisance'], dtype=float),
                      np.column_stack((design, y)), float(study['nuisance_ridge']))
    s, r = values[:, :-1], values[:, -1]
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
        certs.append({'weights': w.copy(), 'nu_sum': nu, 'lam_lower': lower,
                      'lam_upper': upper, 'primal_objective': float(primal),
                      'dual_lower_bound': float(dual)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': (terminal @ weights[1]).reshape(study['nx'], study['ny']),
            'kkt_certificates': certs}
