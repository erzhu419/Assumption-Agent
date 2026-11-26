"""Separable tracer transport, bank calibration, and box-simplex inversion."""
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


def _bank_fit(x, y, historical_gain):
    """Fit an affine response, trimming isolated sweep spikes before refitting."""
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    distinct = np.abs(dx) > 1e-8 * max(1.0, float(np.max(np.abs(x))))
    if not np.any(distinct):
        return historical_gain, float(np.median(y - historical_gain * x))
    if x.size >= 4:
        slopes = [np.median(dy[i, row] / dx[i, row])
                  for i, row in enumerate(distinct) if np.any(row)]
        slope = np.median(slopes)
        residual = y - slope * x
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        keep = np.abs(residual - center) <= max(6.0 * mad, 1e-7)
        if np.any(distinct[np.ix_(keep, keep)]):
            x, y = x[keep], y[keep]
    x0, y0 = x.mean(), y.mean()
    centered = x - x0
    slope = float(centered @ (y - y0) / (centered @ centered))
    return slope, float(y0 - slope * x0)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    b = np.asarray(sweeps['bank_id'])
    x, y = np.asarray(sweeps['ref_signal']), np.asarray(sweeps['raw_readout'])
    # Both archived campaigns gave unit gain and zero offset in every bank.
    gain, offset = np.ones(4), np.zeros(4)
    for bank in range(4):
        take = b == bank
        if np.any(take):
            gain[bank], offset[bank] = _bank_fit(x[take], y[take], gain[bank])
    row_bank = np.asarray(study['row_bank_id'])
    return (np.asarray(study['y_obs_raw']) - offset[row_bank]) / gain[row_bank]


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


def solve_beta_w02_batch_02(study: dict) -> dict:
    """Reconstruct the terminal field and certify the fit at each budget."""
    design, last_fields = _forward(study)
    S, y = _profile(study, design, _calibrate(study))
    ridge, cap = (float(study[k]) for k in ('source_ridge', 'u_box'))
    H = S.T @ S + ridge * np.eye(S.shape[1])
    c = -(S.T @ y)
    weights, certificates = [], []
    for budget in map(float, study['budget_grid']):
        w, nu = _box_simplex(H, c, cap, budget)
        residual = S @ w - y
        grad = S.T @ residual + ridge * w
        reduced = grad - nu
        primal = 0.5 * float(residual @ residual + ridge * (w @ w))
        # Minimize the tangent affine objective on the box after dualizing the sum.
        lower = primal - float(grad @ w) + nu * budget + cap * float(np.minimum(reduced, 0).sum())
        weights.append(w)
        certificates.append({'weights': w.copy(), 'nu_sum': nu,
                             'lam_lower': np.maximum(reduced, 0),
                             'lam_upper': np.maximum(-reduced, 0),
                             'primal_objective': primal,
                             'dual_lower_bound': min(primal, lower)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], last_fields),
            'kkt_certificates': certificates}
