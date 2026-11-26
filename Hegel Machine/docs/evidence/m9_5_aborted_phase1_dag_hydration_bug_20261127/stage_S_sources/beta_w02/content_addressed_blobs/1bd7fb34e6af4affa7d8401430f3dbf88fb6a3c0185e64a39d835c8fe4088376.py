"""Finite-volume tracer tomography with an affine sensor-bank calibration."""
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    kappa = np.asarray(kappa, dtype=float)
    vel = np.asarray(vel, dtype=float)
    d = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    forward = np.maximum(vel, 0) / h
    backward = np.maximum(-vel, 0) / h
    main = np.array(diag_add, dtype=float, copy=True) + forward + backward
    main[:-1] += d
    main[1:] += d
    for i in (0, n - 1):
        main[i] += robin * kappa[i] / (h * (robin * h / 2 + kappa[i]))
    return sparse.diags((-d - forward[1:], main, -d - backward[:-1]), (-1, 0, 1), format='csc')


def _responses(study):
    st = study
    nx, ny = st['nx'], st['ny']
    reaction = st['V_reaction_2d']
    # The supplied axis reactions are part of the base operators; V adds
    # a separate, separable reaction potential.
    vx = st['reaction_x'] + reaction.mean(axis=1)
    vy = st['reaction_y'] + reaction.mean(axis=0) - reaction.mean()
    ax = _build_1d(st['Lx'], nx, st['kappa_x'], st['vel_x'], st['robin_x'], vx)
    by = _build_1d(st['Ly'], ny, st['kappa_y'], st['vel_y'], st['robin_y'], vy)
    A = sparse.kron(ax, sparse.eye(ny), format='csc') + sparse.kron(sparse.eye(nx), by, format='csc')
    I = sparse.eye(nx * ny, format='csc')
    rhs = np.asarray(st['source_modes']).reshape(-1, nx * ny).T
    i, j = np.asarray(st['sensor_ij']).T
    blocks = []
    for mu in st['mu_shifts']:
        fields = splu(A + mu * I).solve(rhs).reshape(nx, ny, -1)
        blocks.append(fields[i, j])
    return np.concatenate(blocks), fields


def _calibration(study):
    sweeps = study['sensor_calibration_sweeps']
    bank, x, y = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    fits = {}
    for b in range(4):
        xb, yb = x[bank == b], y[bank == b]
        if np.unique(xb).size < 2:
            continue
        # A true line has a majority consensus even when a sweep has EMI spikes.
        design = np.column_stack((xb, np.ones_like(xb)))
        pairs = np.asarray(list(combinations(range(len(xb)), 2)))
        i, j = pairs.T
        good = np.abs(xb[j] - xb[i]) > 1e-12
        i, j = i[good], j[good]
        slopes = (yb[j] - yb[i]) / (xb[j] - xb[i])
        offsets = yb[i] - slopes * xb[i]
        residuals = np.abs(design @ np.stack((slopes, offsets)) - yb[:, None])
        best = np.argmin(np.median(residuals, axis=0))
        err = residuals[:, best]
        keep = err <= max(1e-10, 4 * np.median(err))
        fits[b] = np.linalg.lstsq(design[keep], yb[keep], rcond=None)[0]
    for b in range(4):
        if b in fits:
            continue
        neighbors = [fits[k] for k in (b ^ 1,) if k in fits]
        gain, offset = np.median(neighbors or list(fits.values()) or [(1., 0.)], axis=0)
        use = bank == b
        if use.any():
            offset = np.median(y[use] - gain * x[use])
        fits[b] = gain, offset
    return np.array([fits[b] for b in range(4)]).T


def _profile(S, y, Z, ridge):
    data = np.column_stack((S, y))
    q, d, _ = np.linalg.svd(Z, full_matrices=False)
    factor = np.divide(d*d, d*d + ridge, out=np.zeros_like(d), where=d*d + ridge > 0)
    data -= q @ (factor[:, None] * (q.T @ data))
    return data[:, :-1], data[:, -1]


def _box_simplex(S, y, ridge, upper, budget):
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    c = -S.T @ y
    w = np.full(m, budget / m)
    status = np.zeros(m, dtype=int)
    if budget == 0:
        return np.zeros(m), float(c.min())
    if budget == m * upper:
        w = np.full(m, upper)
        return w, float((H @ w + c).max())
    for _ in range(20 * m):
        free = status == 0
        if not free.any():
            g = H @ w + c
            lo = np.argmin(np.where(status == -1, g, np.inf))
            hi = np.argmax(np.where(status == 1, g, -np.inf))
            if g[hi] <= g[lo] + 1e-12:
                return w, float((g[lo] + g[hi]) / 2)
            status[[lo, hi]] = 0
            continue
        fixed = ~free
        k = free.sum()
        mat = np.empty((k+1, k+1))
        mat[:-1, :-1] = H[np.ix_(free, free)]
        mat[:-1, -1] = -1
        mat[-1, :-1] = 1
        mat[-1, -1] = 0
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        x, nu = sol[:-1], sol[-1]
        ids = np.flatnonzero(free)
        p = x - w[ids]
        hit = (x < -1e-12) | (x > upper + 1e-12)
        if hit.any():
            hids = ids[hit]
            bound = np.where(x[hit] < 0, 0., upper)
            steps = (bound - w[hids]) / p[hit]
            a = int(np.argmin(steps))
            w[ids] += np.clip(steps[a], 0., 1.) * p
            w[hids[a]] = bound[a]
            status[hids[a]] = -1 if bound[a] == 0 else 1
            continue
        w[ids] = np.clip(x, 0., upper)
        g = H @ w + c
        bad = status * (g - nu)
        i = np.argmax(bad)
        if bad[i] <= 1e-12:
            return w, float(nu)
        status[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_w02_s00(study: dict) -> dict:
    S, fields = _responses(study)
    gain, offset = _calibration(study)
    banks = np.asarray(study['row_bank_id'], dtype=int)
    y = (study['y_obs_raw'] - offset[banks]) / gain[banks]
    S, y = _profile(S, y, study['Z_nuisance'], study['nuisance_ridge'])
    ridge, upper, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, y, ridge, upper, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    lower = min(J, float(J - g @ w + nu * budget + upper * np.minimum(0., g - nu).sum()))
    return {'weights': w, 'terminal_field': fields @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., g - nu),
                                'lam_upper': np.maximum(0., nu - g),
                                'primal_objective': J, 'dual_lower_bound': lower}}
