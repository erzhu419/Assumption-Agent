"""Separable finite-volume tomography with bank calibration and a box-simplex certificate."""

import numpy as np


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    diffusion = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    forward = np.maximum(vel, 0) / h
    backward = np.maximum(-vel, 0) / h
    diagonal = np.r_[0, diffusion] + np.r_[diffusion, 0] + forward + backward + diag_add
    edge = np.array([0, n - 1])
    diagonal[edge] += robin * kappa[edge] / (h * (0.5 * robin * h + kappa[edge]))
    return (np.diag(diagonal) - np.diag(diffusion + backward[:-1], 1)
            - np.diag(diffusion + forward[1:], -1))


def _axis_modes(operator):
    # D^-1 A D is symmetric: upwind changes the two off-diagonals differently.
    upper = np.diag(operator, 1)
    lower = np.diag(operator, -1)
    off = -np.sqrt(upper * lower)
    scale = np.r_[1., np.cumprod(np.sqrt(lower / upper))]
    modes, vectors = np.linalg.eigh(np.diag(np.diag(operator)) + np.diag(off, 1) + np.diag(off, -1))
    return modes, scale[:, None] * vectors, vectors.T / scale


def _responses(study):
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    rx = v.mean(axis=1) + np.asarray(study['reaction_x'])
    ry = v.mean(axis=0) - v.mean() + np.asarray(study['reaction_y'])
    axes = []
    for axis, reaction in zip('xy', (rx, ry)):
        axes.append(_axis_modes(_build_1d(
            study['L' + axis], study['n' + axis], np.asarray(study['kappa_' + axis]),
            np.asarray(study['vel_' + axis]), study['robin_' + axis], reaction)))
    (ex, px, ix), (ey, py, iy) = axes
    rhs = ix @ np.asarray(study['source_modes']) @ iy.T
    spectrum = ex[:, None] + ey[None, :]
    amplitudes = rhs[None] / (spectrum[None, None] + np.asarray(study['mu_shifts'])[:, None, None, None])
    return px @ amplitudes @ py.T


def _fit_bank(x, y, prior):
    gain, offset = prior
    i, j = np.triu_indices(x.size, 1)
    delta = x[j] - x[i]
    valid = np.abs(delta) > 1e-10
    if not valid.any():
        return gain, float(np.median(y - gain * x)) if x.size else offset
    seed_gain = np.median((y[j[valid]] - y[i[valid]]) / delta[valid])
    residual = y - seed_gain * x
    residual -= np.median(residual)
    keep = np.abs(residual) <= max(1e-7, 5 * np.median(np.abs(residual)))
    if keep.sum() >= 2 and np.ptp(x[keep]) > 1e-10:
        a, b = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))),
                              y[keep], rcond=None)[0]
        if a > 0:
            return float(a), float(b)
    return gain, float(np.median(y - gain * x))


def _calibrate(study):
    import json
    from pathlib import Path

    sweep = study['sensor_calibration_sweeps']
    banks = np.asarray(sweep['bank_id'])
    ref = np.asarray(sweep['ref_signal'], dtype=float)
    raw = np.asarray(sweep['raw_readout'], dtype=float)
    gains = np.ones(4)
    offsets = np.zeros(4)
    history = None
    for bank in range(4):
        mask = banks == bank
        x, y = ref[mask], raw[mask]
        prior = (1., 0.)
        if x.size < 2 or np.ptp(x) <= 1e-10:
            if history is None:
                path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
                history = json.loads(path.read_text()) if path.is_file() else []
            for record in reversed(history):
                hist_mask = np.asarray(record['bank_id']) == bank
                hx = np.asarray(record['ref_signal'])[hist_mask]
                if hx.size >= 2 and np.ptp(hx) > 1e-10:
                    prior = _fit_bank(hx, np.asarray(record['raw_readout'])[hist_mask], prior)
                    break
        gains[bank], offsets[bank] = _fit_bank(x, y, prior)
    row_bank = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw']) - offsets[row_bank]) / gains[row_bank]


def _project(v, cap, budget):
    lo, hi = v.min() - cap, v.max()
    for _ in range(55):
        mid = (lo + hi) / 2
        if np.clip(v - mid, 0, cap).sum() > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(v - (lo + hi) / 2, 0, cap)


def _box_simplex(S, y, ridge, cap, budget):
    m = S.shape[1]
    H, f = S.T @ S + ridge * np.eye(m), S.T @ y
    w = np.full(m, budget / m)
    step = np.linalg.eigvalsh(H)[-1]
    if 0 < budget < m * cap and step > 0:
        z = w.copy()
        for t in range(1, 1001):
            new = _project(z - (H @ z - f) / step, cap, budget)
            gradient = H @ new - f
            if gradient @ new - _linear_minimum(gradient, cap, budget) < 2e-13:
                w = new
                break
            z = new + (t - 1) / (t + 2) * (new - w)
            w = new
        free = (w > 1e-10) & (w < cap - 1e-10)
        if free.any():
            # Solve the equality-constrained subproblem on the identified face.
            upper = w >= cap - 1e-10
            F = H[np.ix_(free, free)]
            ones = np.ones(free.sum())
            kkt = np.block([[F, -ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            rhs = np.r_[f[free] - cap * H[np.ix_(free, upper)].sum(axis=1), budget - cap * upper.sum()]
            exact = np.zeros_like(w)
            exact[upper] = cap
            exact[free] = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
            if np.all((exact >= 0) & (exact <= cap)):
                gradient = H @ exact - f
                if gradient @ exact - _linear_minimum(gradient, cap, budget) < 2e-13:
                    w = exact
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    # A budget quantile maximizes the box-relaxed affine lower bound.
    nu = float(np.sort(g)[min(m - 1, int(budget / cap))]) if cap > 0 else 0.
    ll, lu = np.maximum(g - nu, 0), np.maximum(nu - g, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * lu.sum())
    return w, dict(weights=w.copy(), nu_sum=nu, lam_lower=ll, lam_upper=lu,
                   primal_objective=objective, dual_lower_bound=min(objective, dual))


def _linear_minimum(g, cap, budget):
    fill = np.clip(budget - cap * np.arange(g.size), 0, cap)
    return np.sort(g) @ fill


def solve_beta_w03_batch_03(study: dict) -> dict:
    U = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    S = U[:, :, i, j].transpose(0, 2, 1).reshape(-1, U.shape[1])
    Y = np.column_stack((S, _calibrate(study)))
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    if Z.shape[1]:
        Q, sigma, _ = np.linalg.svd(Z, full_matrices=False)
        squared = sigma**2
        denominator = squared + study['nuisance_ridge']
        shrink = np.divide(squared, denominator, out=np.zeros_like(squared), where=denominator > 0)
        Y -= Q @ (shrink[:, None] * (Q.T @ Y))
    pairs = [_box_simplex(Y[:, :-1], Y[:, -1], study['source_ridge'], study['u_box'], float(b))
             for b in study['budget_grid']]
    weights, certificates = zip(*pairs)
    terminal = np.einsum('m,mxy->xy', weights[1], U[-1])
    return {'weights_by_budget': np.stack(weights), 'terminal_field_mid_budget': terminal,
            'kkt_certificates': list(certificates)}
