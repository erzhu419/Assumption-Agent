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
    if not x.size:
        return gain, offset
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    use = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x)))
    if not use.any():
        return gain, float(np.median(y - gain * x))
    slopes = (y[j[use]] - y[i[use]]) / dx[use]
    slopes = slopes[slopes > 0]
    if not slopes.size:
        return gain, float(np.median(y - gain * x))
    # A majority-consensus affine fit, followed by OLS on the clean sweeps.
    offsets = np.median(y[None, :] - slopes[:, None] * x, axis=1)
    residuals = np.abs(y[None, :] - slopes[:, None] * x - offsets[:, None])
    best = np.argmin(np.median(residuals, axis=1))
    residual = residuals[best]
    clean = residual <= max(1e-7, 5 * np.median(residual))
    if np.any(clean) and np.ptp(x[clean]) > 1e-10:
        gain, offset = np.linalg.lstsq(np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
        if gain > 0:
            return float(gain), float(offset)
    return float(slopes[best]), float(offsets[best])


def _calibrate(study, S, Z):
    sweep = study['sensor_calibration_sweeps']
    ids = np.asarray(sweep['bank_id'])
    ref = np.asarray(sweep['ref_signal'], dtype=float)
    raw = np.asarray(sweep['raw_readout'], dtype=float)
    row = np.asarray(study['row_bank_id'], dtype=int)
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    y = np.empty_like(obs)
    sparse = []
    anchored = False
    for bank in range(4):
        rows = row == bank
        if not rows.any():
            continue
        x, z = ref[ids == bank], raw[ids == bank]
        gain, offset = _fit_bank(x, z, (1., 0.))
        y[rows] = (obs[rows] - offset) / gain
        if x.size and np.ptp(x) > 1e-10 * max(1., np.max(np.abs(x))):
            anchored = True
        else:
            sparse.append((rows, x, z))
    if not sparse or not anchored:
        return y

    # Use calibrated banks to anchor a shared transport/nuisance waveform.
    # A single spot constrains offset conditional on gain; no spot leaves
    # both gain and offset to be estimated from the shared waveform.
    base = y.copy()
    columns = []
    gain_columns = []
    for rows, x, z in sparse:
        anchor = float(np.median(x)) if x.size else 0.
        base[rows] = anchor
        gain_columns.append(len(columns))
        columns.append(np.where(rows, y - anchor, 0.))
        if not z.size:
            columns.append(rows.astype(float))
    Q = np.column_stack(columns)
    m = S.shape[1]
    tangent = np.linalg.qr(np.column_stack((np.ones(m), np.eye(m)[:, :-1])))[0][:, 1:]
    uniform = S.sum(axis=1) * (study['total_budget'] / m)
    model = np.column_stack((S @ tangent, Z, -Q))
    penalty = np.zeros((m - 1, model.shape[1]))
    penalty[:, :m - 1] = np.sqrt(study['source_ridge']) * np.eye(m - 1)
    theta, _, rank, _ = np.linalg.lstsq(np.vstack((model, penalty)),
                                       np.r_[base - uniform, np.zeros(m - 1)], rcond=None)
    if rank == model.shape[1]:
        scales = theta[-Q.shape[1]:]
        if np.all(scales[gain_columns] > 0):
            return base + Q @ scales
    return y


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


def solve_beta_w03_s00(study: dict) -> dict:
    U = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    S = U[:, :, i, j].transpose(0, 2, 1).reshape(-1, U.shape[1])
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrate(study, S, Z)
    data = np.column_stack((S, y))
    # Contract uses the ridge residual-maker, not its square root.
    if Z.shape[1]:
        Q, sigma, _ = np.linalg.svd(Z, full_matrices=False)
        attenuation = np.divide(sigma**2, sigma**2 + study['nuisance_ridge'],
                                out=np.zeros_like(sigma), where=sigma**2 + study['nuisance_ridge'] > 0)
        data -= Q @ (attenuation[:, None] * (Q.T @ data))
    weights, certificate = _box_simplex(data[:, :-1], data[:, -1], study['source_ridge'],
                                       study['u_box'], study['total_budget'])
    return dict(weights=weights, terminal_field=np.einsum('m,mxy->xy', weights, U[-1]),
                kkt_certificate=certificate)
