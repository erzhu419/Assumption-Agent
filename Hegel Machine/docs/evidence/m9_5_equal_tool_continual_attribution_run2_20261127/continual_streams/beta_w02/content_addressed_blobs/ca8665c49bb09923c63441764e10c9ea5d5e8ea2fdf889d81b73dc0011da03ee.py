"""Bank-calibrated inversion for the vortex tracer survey."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    """Return the three diagonals of the cell-centered FVM operator."""
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(vel, dtype=float) / h
    diffusion = (2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    diag = np.array(diag_add, dtype=float, copy=True) + np.abs(v)
    diag[:-1] += diffusion
    diag[1:] += diffusion
    boundary = k[[0, -1]]
    diag[[0, -1]] += robin * boundary / (h * (robin * 0.5 * h + boundary))
    return -diffusion - np.maximum(v[1:], 0), diag, -diffusion - np.maximum(-v[:-1], 0)


def _forward(study):
    """Solve each shifted 2D system for all source modes at once."""
    nx, ny = study['nx'], study['ny']
    V = np.asarray(study['V_reaction_2d'])
    axes = []
    for axis, n in (('x', nx), ('y', ny)):
        diagonals = _build_1d(
            study['L' + axis], n, 1.45 * np.asarray(study['kappa_' + axis]),
            study['vel_' + axis], study['robin_' + axis],
            study['reaction_' + axis],
        )
        axes.append(diags(diagonals, (-1, 0, 1), format='csc'))
    ident = eye(nx * ny, format='csc')
    base = (kron(axes[0], eye(ny), format='csc')
            + kron(eye(nx), axes[1], format='csc')
            + diags(V.ravel(), format='csc'))
    sources = np.asarray(study['source_modes']).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    rows = i * ny + j
    blocks = []
    for shift in study['mu_shifts']:
        solution = splu(base + shift * ident).solve(sources)
        blocks.append(solution[rows])
    return np.concatenate(blocks, axis=0), solution.T.reshape(-1, nx, ny)


def _fit_line(x, y):
    X = np.column_stack((x, np.ones(x.size)))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _sweep_columns(sweep):
    return tuple(np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))


def _identify_line(x, y):
    # Use a strict majority rather than fitting a minority of EMI readings.
    minimum = max(2, x.size // 2 + 1)
    if x.size < minimum or np.ptp(x) <= 1e-8:
        return None
    p = _fit_line(x, y)
    tol = 3e-6  # Includes the archive's six-decimal sweep precision.
    if p[0] > 0 and np.max(np.abs(y - p[0] * x - p[1])) <= tol:
        return p
    best = None
    for i, j in combinations(range(x.size), 2):
        dx = x[j] - x[i]
        if abs(dx) <= 1e-8:
            continue
        gain = (y[j] - y[i]) / dx
        if gain <= 0:
            continue
        residual = np.abs(y - gain * (x - x[i]) - y[i])
        use = residual <= tol
        count = int(use.sum())
        # With three inconsistent checks, any pair can define a line.
        if count < max(3, minimum):
            continue
        score = (count, -float(residual[use] @ residual[use]))
        if best is None or score > best[0]:
            best = score, use
    return None if best is None else _fit_line(x[best[1]], y[best[1]])


def _calibrate(study):
    bank, x, y = _sweep_columns(study['sensor_calibration_sweeps'])
    gain, offset = np.ones(4), np.zeros(4)
    history = None
    rows = np.asarray(study['row_bank_id'])
    for b in np.unique(rows):
        use = bank == b
        # Two points always define a line and cannot detect EMI. Wing B's
        # sparse spot checks need a third point to establish a new calibration.
        params = None
        if b < 2 or np.count_nonzero(use) >= 3:
            params = _identify_line(x[use], y[use])
        if params is None:
            if history is None:
                history = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
            for episode in reversed(history):
                hb, hx, hy = _sweep_columns(episode)
                keep = hb == b
                params = _identify_line(hx[keep], hy[keep])
                if params is not None:
                    break
            if params is None:
                raise ValueError(f'No identifiable calibration for bank {b}')
        gain[b], offset[b] = params
    return (np.asarray(study['y_obs_raw']) - offset[rows]) / gain[rows]


def _profile(study, design, y):
    Z = np.asarray(study['Z_nuisance'])
    stacked = np.column_stack((design, y))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residual = stacked - Z @ np.linalg.solve(gram, Z.T @ stacked)
    return residual[:, :-1], residual[:, -1]


def _box_simplex(H, c, cap, budget):
    """Feasible active-set solve; nu follows the validator's g - nu convention."""
    m = len(c)
    if budget == 0:
        return np.zeros(m), float(np.min(c))
    if budget == m * cap:
        w = np.full(m, cap)
        return w, float(np.max(H @ w + c))
    w = np.full(m, budget / m)
    active = np.zeros(m, dtype=int)  # -1 lower, +1 upper
    for _ in range(20 * m * m):
        free = np.flatnonzero(active == 0)
        grad = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(active == -1), np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[i] >= grad[j] - 1e-12:
                return w, float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue
        system = np.empty((free.size + 1, free.size + 1))
        system[:-1, :-1] = H[np.ix_(free, free)]
        system[:-1, -1] = -1
        system[-1, :-1] = 1
        system[-1, -1] = 0
        step = np.linalg.solve(system, np.r_[-grad[free], budget - w.sum()])
        move, nu = step[:-1], step[-1]
        limit = np.full(free.size, np.inf)
        up, down = move > 1e-13, move < -1e-13
        limit[up] = (cap - w[free[up]]) / move[up]
        limit[down] = -w[free[down]] / move[down]
        k = np.argmin(limit)
        if limit[k] < 1.0:
            w[free] += max(0.0, limit[k]) * move
            idx = free[k]
            active[idx] = 1 if up[k] else -1
            w[idx] = cap if up[k] else 0.0
            continue
        w[free] += move
        grad = H @ w + c
        violation = active * (grad - nu)
        k = np.argmax(violation)
        if violation[k] <= 1e-12:
            return np.clip(w, 0, cap), float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_w02_batch_08(study: dict) -> dict:
    design, last = _forward(study)
    S, y = _profile(study, design, _calibrate(study))
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -(S.T @ y), cap, budget)
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    objective = 0.5 * float(residual @ residual + ridge * (w @ w))
    reduced = gradient - nu
    lower_bound = objective - float(gradient @ w) + nu * budget + cap * float(np.minimum(reduced, 0).sum())
    return {'weights': w, 'terminal_field': np.einsum('m,mij->ij', w, last),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(reduced, 0),
                                'lam_upper': np.maximum(-reduced, 0),
                                'primal_objective': objective, 'dual_lower_bound': lower_bound}}
