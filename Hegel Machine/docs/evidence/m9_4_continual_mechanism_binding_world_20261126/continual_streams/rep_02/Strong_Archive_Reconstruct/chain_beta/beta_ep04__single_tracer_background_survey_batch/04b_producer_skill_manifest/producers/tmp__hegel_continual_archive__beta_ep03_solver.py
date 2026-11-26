"""Calibrated dual-tracer transport and certified source reconstruction."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    banks = np.asarray(study['row_bank_id'])
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    y = np.empty_like(raw)
    for bank in np.unique(banks):
        mask = np.asarray(sweep['bank_id']) == bank
        ref = np.asarray(sweep['ref_signal'])[mask]
        readout = np.asarray(sweep['raw_readout'])[mask]
        gain, offset = np.linalg.lstsq(
            np.column_stack((ref, np.ones_like(ref))), readout, rcond=None
        )[0]
        rows = banks == bank
        y[rows] = (raw[rows] - offset) / gain
    return y


def _axis(study, axis):
    n = study['n' + axis]
    h = study['L' + axis] / n
    k = np.asarray(study['kappa_' + axis])
    v = np.asarray(study['vel_' + axis]) / h
    rg = study['robin_' + axis]
    faces = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    diagonal = np.asarray(study['reaction_' + axis]) + np.abs(v)
    diagonal[:-1] += faces
    diagonal[1:] += faces
    k_end = k[[0, -1]]
    diagonal[[0, -1]] += rg * k_end / (h * (0.5 * rg * h + k_end))
    return sparse.diags(
        (-faces - np.maximum(v[1:], 0), diagonal, -faces + np.minimum(v[:-1], 0)),
        (-1, 0, 1), format='csc',
    )


def _transport(study):
    modes = np.asarray(study['source_modes'])
    m, nx, ny = modes.shape
    rhs = modes.reshape(m, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    rows = i * ny + j
    ax, by = _axis(study, 'x'), _axis(study, 'y')
    ix, iy = sparse.eye(nx, format='csc'), sparse.eye(ny, format='csc')
    eye = sparse.eye(nx * ny, format='csc')
    va = np.asarray(study['V_channel_a'])
    axa = ax + sparse.diags(va[:, 0] - va[0, 0])
    bya = by + sparse.diags(va[0, :])
    a = sparse.kron(axa, iy, format='csc') + sparse.kron(ix, bya, format='csc')
    b = sparse.kron(ax, iy, format='csc') + sparse.kron(ix, by, format='csc')
    b += sparse.diags(np.asarray(study['V_channel_b']).ravel(), format='csc')
    blocks, terminal = [], []
    for op in (a, b):
        for mu in study['mu_shifts']:
            field = splu(op + mu * eye).solve(rhs)
            blocks.append(field[rows])
        terminal.append(field)
    return np.vstack(blocks), terminal


def _profile(design, y, study):
    z = np.asarray(study['Z_nuisance'])
    both = np.column_stack((design, y))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    detrended = both - z @ np.linalg.solve(gram, z.T @ both)
    return detrended[:, :-1], detrended[:, -1]


def _box_simplex(h, linear, cap, budget):
    """Find the optimum by equality-constrained solves on successive faces."""
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = h @ w + linear
        return w, float(g.min() if budget == 0 else g.max())

    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1 at zero, +1 at cap
    for _ in range(100 * n):
        free = np.flatnonzero(active == 0)
        g = h @ w + linear
        if not len(free):
            lo = np.flatnonzero(active < 0)
            hi = np.flatnonzero(active > 0)
            low, high = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[high] <= g[low] + 1e-12:
                return w, float((g[high] + g[low]) / 2)
            active[[low, high]] = 0
            continue

        gf = g[free]
        a, b = np.linalg.solve(
            h[np.ix_(free, free)], np.column_stack((gf, np.ones(len(free))))
        ).T
        remaining = budget - float(w.sum())
        nu = float((a.sum() + remaining) / b.sum())
        step = nu * b - a
        trial = w[free] + step
        hit = (trial < -1e-13) | (trial > cap + 1e-13)
        if np.any(hit):
            candidates = np.flatnonzero(hit)
            bounds = np.where(step[candidates] < 0, 0.0, cap)
            fractions = (bounds - w[free[candidates]]) / step[candidates]
            k = int(np.argmin(fractions))
            w[free] = np.clip(w[free] + fractions[k] * step, 0, cap)
            idx = free[candidates[k]]
            w[idx] = bounds[k]
            active[idx] = -1 if bounds[k] == 0 else 1
            continue

        w[free] = np.clip(trial, 0, cap)
        g = h @ w + linear
        nu = float(g[free].mean())
        violation = active * (g - nu)
        idx = int(np.argmax(violation))
        if violation[idx] <= 1e-12:
            return w, nu
        active[idx] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_ep03(study: dict) -> dict:
    """Return source weights, both last-shift fields, and a KKT certificate."""
    design, fields = _transport(study)
    s, y = _profile(design, _calibrate(study), study)
    ridge, cap, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    h = s.T @ s + ridge * np.eye(s.shape[1])
    w, nu = _box_simplex(h, -s.T @ y, cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    lower = np.maximum(g - nu, 0.0)
    upper = np.maximum(nu - g, 0.0)
    primal = float((residual @ residual + ridge * (w @ w)) / 2)
    dual = primal - float(lower @ w + upper @ (cap - w)) + nu * (budget - float(w.sum()))
    certificate = dict(weights=w.tolist(), nu_sum=nu, lam_lower=lower.tolist(),
                       lam_upper=upper.tolist(), primal_objective=primal,
                       dual_lower_bound=dual)
    shape = (study['nx'], study['ny'])
    return dict(weights=w, terminal_field_channel_a=(fields[0] @ w).reshape(shape),
                terminal_field_channel_b=(fields[1] @ w).reshape(shape),
                kkt_certificate=certificate)
