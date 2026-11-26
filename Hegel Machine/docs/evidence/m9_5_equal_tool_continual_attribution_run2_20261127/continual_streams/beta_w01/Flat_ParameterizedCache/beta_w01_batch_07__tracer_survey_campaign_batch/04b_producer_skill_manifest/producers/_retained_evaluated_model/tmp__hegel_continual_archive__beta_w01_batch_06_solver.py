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


def _fit_bank(x, y, prior=(1., 0.)):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    independent = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x), initial=0.))
    if not independent.any():
        gain, offset = prior
        return gain, float(np.median(y - gain * x)) if len(x) else offset
    gain = float(np.median((y[j[independent]] - y[i[independent]]) / dx[independent]))
    residual = y - gain * x
    offset = float(np.median(residual))
    deviation = np.abs(residual - offset)
    scale = 1.4826 * np.median(deviation)
    keep = deviation <= max(4.5 * scale, 1e-10)
    if np.ptp(x[keep]) > 0 if keep.any() else False:
        gain, offset = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
    return (gain, offset) if gain > 0 else prior


# Archived affine fits for the regime matching the current bank-0 sweep.
_HISTORICAL_BANKS = ((0.7697, -0.0964), (1.1935, 0.1420), (0.7170, -0.0972), (1.1920, 0.0807))


def _calibrated_observations(study):
    sweep = study['sensor_calibration_sweeps']
    bank = np.asarray(sweep['bank_id']).ravel()
    ref = np.asarray(sweep['ref_signal'], dtype=float).ravel()
    raw = np.asarray(sweep['raw_readout'], dtype=float).ravel()
    rows = np.asarray(study['row_bank_id'], dtype=int).ravel()
    observed = np.asarray(study['y_obs_raw'], dtype=float).ravel()
    gains, offsets = np.transpose([
        _fit_bank(ref[bank == b], raw[bank == b], prior)
        for b, prior in enumerate(_HISTORICAL_BANKS)
    ])
    return (observed - offsets[rows]) / gains[rows]


def _responses(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    ax = _build_1d(study['Lx'], nx, 1.45 * np.asarray(study['kappa_x']),
                   study['vel_x'], study['robin_x'], study['reaction_x'])
    by = _build_1d(study['Ly'], ny, 1.45 * np.asarray(study['kappa_y']),
                   study['vel_y'], study['robin_y'], study['reaction_y'])
    operator = (kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
                + diags(np.asarray(study['V_reaction_2d']).ravel(), format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = sources.reshape(m, -1).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    rows = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs).reshape(nx, ny, m)
        rows.append(fields[ix, iy])
    return np.vstack(rows), fields


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


def solve_beta_w01_batch_06(study: dict) -> dict:
    design, terminal = _responses(study)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrated_observations(study)
    profiled = _profile(nuisance,
                        np.column_stack((design, y)), float(study['nuisance_ridge']))
    s, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    hessian = s.T @ s + ridge * np.eye(s.shape[1])
    w, nu = _box_simplex(hessian, -(s.T @ r), cap, budget)
    residual = s @ w - r
    g = s.T @ residual + ridge * w
    lower, upper = np.maximum(g - nu, 0.), np.maximum(nu - g, 0.)
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = primal - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': terminal @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': float(primal), 'dual_lower_bound': float(dual)}}
