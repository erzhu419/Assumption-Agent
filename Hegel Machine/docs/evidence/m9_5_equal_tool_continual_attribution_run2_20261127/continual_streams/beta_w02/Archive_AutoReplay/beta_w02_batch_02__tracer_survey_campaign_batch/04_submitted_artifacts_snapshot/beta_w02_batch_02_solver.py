"""Solve the multi-wing tracer survey and certify all source budgets."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal


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


def _spectral_axis(study, axis, reaction):
    lo, mid, hi = _build_1d(
        study['L' + axis], study['n' + axis], study['kappa_' + axis],
        study['vel_' + axis], study['robin_' + axis],
        study['reaction_' + axis] + reaction,
    )
    eig, orthogonal = eigh_tridiagonal(mid, -np.sqrt(lo * hi))
    # Diagonal similarity reduces the nonsymmetric tridiagonal to a symmetric one.
    scale = np.r_[1.0, np.cumprod(np.sqrt(lo / hi))]
    return eig, scale[:, None] * orthogonal, orthogonal.T / scale


def _forward(study):
    reaction = np.asarray(study['V_reaction_2d'])
    ax, X, Xi = _spectral_axis(study, 'x', reaction.mean(axis=1))
    ay, Y, Yi = _spectral_axis(study, 'y', reaction.mean(axis=0) - reaction.mean())
    modal_sources = Xi @ study['source_modes'] @ Yi.T
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for mu in study['mu_shifts']:
        fields = X @ (modal_sources / (ax[:, None] + ay[None, :] + mu)) @ Y.T
        blocks.append(fields[:, i, j].T)
    return np.concatenate(blocks, axis=0), fields


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


def _affine_sweep(x, y):
    """Fit on clean references after screening isolated readout spikes."""
    i, j = np.triu_indices(x.size, 1)
    keep = np.abs(x[j] - x[i]) > 1e-8 * max(1.0, np.max(np.abs(x), initial=0.0))
    i, j = i[keep], j[keep]
    if not i.size:
        return None
    if x.size >= 4:
        slopes = (y[j] - y[i]) / (x[j] - x[i])
        intercepts = y[i] - slopes * x[i]
        errors = np.abs(y[None, :] - slopes[:, None] * x - intercepts[:, None])
        # Minimize the majority residual; the most widely separated pair
        # breaks near-ties when many pairs agree at reference precision.
        k = x.size // 2
        scores = np.partition(errors, k, axis=1)[:, k]
        close = scores <= max(1e-7, 1.5 * scores.min())
        line = np.argmax(np.where(close, np.abs(x[j] - x[i]), -1.0))
        residual = y - slopes[line] * x
        deviation = np.abs(residual - np.median(residual))
        good = deviation <= max(1e-7, 6.0 * np.median(deviation))
        if np.any(good[i] & good[j]):
            x, y = x[good], y[good]
    xc = x - x.mean()
    gain = float(xc @ (y - y.mean()) / (xc @ xc))
    return gain, float(y.mean() - gain * x.mean())


def _bank_points(sweep, bank):
    sel = np.asarray(sweep['bank_id']) == bank
    return (np.asarray(sweep['ref_signal'], dtype=float)[sel],
            np.asarray(sweep['raw_readout'], dtype=float)[sel])


def _calibrate(study):
    bank_ids = np.asarray(study['row_bank_id'])
    sweep = study['sensor_calibration_sweeps']
    history = None
    calibrated = np.empty(bank_ids.shape, dtype=float)
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    for bank in np.unique(bank_ids):
        x, y = _bank_points(sweep, bank)
        fit = _affine_sweep(x, y)
        if fit is None:
            if history is None:
                path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
                history = json.loads(path.read_text()) if path.is_file() else []
            fit = (1.0, 0.0)
            for old_sweep in reversed(history):
                old_fit = _affine_sweep(*_bank_points(old_sweep, bank))
                if old_fit is not None:
                    fit = old_fit
                    break
            if x.size:
                fit = (fit[0], float(np.median(y - fit[0] * x)))
        rows = bank_ids == bank
        calibrated[rows] = (raw[rows] - fit[1]) / fit[0]
    return calibrated


def solve_beta_w02_batch_02(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    S, y = _profile(study, design, _calibrate(study))
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    c = -(S.T @ y)
    weights, certificates = [], []
    for budget in study['budget_grid']:
        budget = float(budget)
        w, nu = _box_simplex(H, c, cap, budget)
        residual = S @ w - y
        g = S.T @ residual + ridge * w
        reduced = g - nu
        objective = 0.5 * float(residual @ residual + ridge * (w @ w))
        lower = objective - float(g @ w) + nu * budget + cap * float(np.minimum(reduced, 0).sum())
        certificates.append({'weights': w.tolist(), 'nu_sum': nu,
                             'lam_lower': np.maximum(reduced, 0).tolist(),
                             'lam_upper': np.maximum(-reduced, 0).tolist(),
                             'primal_objective': objective, 'dual_lower_bound': lower})
        weights.append(w)
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal_modes),
            'kkt_certificates': certificates}
