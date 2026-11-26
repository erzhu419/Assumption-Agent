"""Calibrated multi-shift tracer tomography with a box-simplex certificate."""

import json
from itertools import product
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


def _historical_bank_fits():
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    fits = [(1., 0.)] * 4
    if path.is_file():
        for sweep in json.loads(path.read_text()):
            banks = np.asarray(sweep['bank_id'])
            ref = np.asarray(sweep['ref_signal'], dtype=float)
            raw = np.asarray(sweep['raw_readout'], dtype=float)
            for bank in range(4):
                use = (banks == bank) & np.isfinite(ref) & np.isfinite(raw)
                if use.any() and np.ptp(ref[use]) > 1e-10:
                    fits[bank] = _fit_bank(ref[use], raw[use], fits[bank])
    return fits


def _calibrated_observations(study, design, nuisance):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    rows = np.asarray(study['row_bank_id'], dtype=int).ravel()
    observed = np.asarray(study['y_obs_raw'], dtype=float).ravel()
    y = observed.copy()
    sparse = []
    short_sweeps = []
    priors = None
    for bank in range(4):
        use = (banks == bank) & np.isfinite(ref) & np.isfinite(raw)
        x, r = ref[use], raw[use]
        take = rows == bank
        if len(x) >= 2 and np.ptp(x) > 1e-10:
            gain, offset = _fit_bank(x, r)
            if take.any() and 3 <= len(x) <= 4 and np.ptp(r - gain * x) > 4e-6:
                # With one spike, at least one of these pairs still gives the
                # clean affine line. The short sweep alone cannot select it.
                i, j = np.triu_indices(len(x), 1)
                dx, dr = x[j] - x[i], r[j] - r[i]
                good = (np.abs(dx) > 1e-10) & (dx * dr > 0)
                slopes = dr[good] / dx[good]
                intercepts = r[i[good]] - slopes * x[i[good]]
                if slopes.size:
                    short_sweeps.append((take, slopes, intercepts))
        else:
            if priors is None:
                priors = _historical_bank_fits()
            gain, offset = priors[bank]
            if take.any() and len(x):
                xb, rb = float(np.median(x)), float(np.median(r))
                # A compatible spot verifies continuity but does not identify a new slope.
                if abs(rb - (gain * xb + offset)) > 2e-6:
                    sparse.append((take, xb, rb, gain))
        y[take] = (observed[take] - offset) / gain
    if not sparse and not short_sweeps:
        return y

    # For disagreeing spots, remove the shared source and nuisance subspace
    # before deciding whether the archived calibration needs to change.
    physics = np.column_stack((design, nuisance))
    q, singular, _ = np.linalg.svd(physics, full_matrices=False)
    q = q[:, singular > np.finfo(float).eps * max(physics.shape) * singular[0]]

    def project(values):
        return values - q @ (q.T @ values)

    if short_sweeps:
        baseline = y.copy()
        shifts = []
        for take, slopes, intercepts in short_sweeps:
            corrected = (observed[take, None] - intercepts) / slopes
            shift = np.zeros((len(y), len(slopes)))
            shift[take] = corrected - baseline[take, None]
            shifts.append(shift)
        residual = project(baseline)
        projected = [project(shift).T for shift in shifts]
        best_loss = np.inf
        best = None
        for indices in product(*(range(len(group)) for group in projected)):
            trial = residual + sum(group[i] for group, i in zip(projected, indices))
            loss = trial @ trial
            if loss < best_loss:
                best_loss, best = loss, indices
        y = baseline + sum(group[:, i] for group, i in zip(shifts, best))
    if not sparse:
        return y

    base = y.copy()
    delta = np.zeros((len(y), len(sparse)))
    for j, (take, xb, rb, gain) in enumerate(sparse):
        base[take] = xb + (observed[take] - rb) / gain
        delta[take, j] = observed[take] - rb
    unchanged_residual = project(y)
    anchored_residual = project(base)
    projected_delta = project(delta)
    correction, _, rank, _ = np.linalg.lstsq(projected_delta, -anchored_residual, rcond=None)
    # Require improvement beyond what a single extra coefficient would fit from noise.
    dof = max(1, len(y) - q.shape[1] - rank)
    corrected_residual = anchored_residual + projected_delta @ correction
    loss = corrected_residual @ corrected_residual
    threshold = 9 * loss / dof
    if rank == len(sparse) and np.all(correction + [1 / row[3] for row in sparse] > 0):
        if min(unchanged_residual @ unchanged_residual, anchored_residual @ anchored_residual) - loss > threshold * rank:
            return base + delta @ correction
    if unchanged_residual @ unchanged_residual - anchored_residual @ anchored_residual > threshold * len(sparse):
        return base
    return y


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


def solve_beta_w01_batch_01(study: dict) -> dict:
    """Recover one set of source weights and certificate per campaign budget."""
    budgets = np.asarray(study['budget_grid'], dtype=float)
    design, terminal = _responses(study)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrated_observations(study, design, nuisance)
    profiled = _profile(nuisance, np.column_stack((design, y)),
                        float(study['nuisance_ridge']))
    s, r = profiled[:, :-1], profiled[:, -1]
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = s.T @ s + ridge * np.eye(s.shape[1])
    linear = -(s.T @ r)
    weights, certs = [], []
    for budget in budgets:
        w, nu = _box_simplex(hessian, linear, cap, float(budget))
        residual = s @ w - r
        g = s.T @ residual + ridge * w
        lower = np.maximum(g - nu, 0.)
        upper = np.maximum(nu - g, 0.)
        primal = 0.5 * (residual @ residual + ridge * (w @ w))
        dual = primal - g @ w + nu * budget - cap * upper.sum()
        weights.append(w)
        certs.append({'weights': w.copy(), 'nu_sum': nu,
                      'lam_lower': lower, 'lam_upper': upper,
                      'primal_objective': float(primal),
                      'dual_lower_bound': float(dual)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': (terminal @ weights[1]).reshape(study['nx'], study['ny']),
            'kkt_certificates': certs}
