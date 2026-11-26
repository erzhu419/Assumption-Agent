"""Calibrated separable reaction-transport inversion."""

from itertools import product

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.broadcast_to(vel, (n,)) / h
    diff = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    # The upwind difference uses the velocity of the receiving cell.
    main = np.asarray(diag_add, dtype=float) + np.abs(v)
    main[:-1] += diff
    main[1:] += diff
    edges = np.array([0, n - 1])
    main[edges] += robin * k[edges] / (h * (0.5 * robin * h + k[edges]))
    lower = -diff - np.maximum(v[1:], 0)
    upper = -diff + np.minimum(v[:-1], 0)
    return sparse.diags((lower, main, upper), (-1, 0, 1), format='csc')


def _transport(study):
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    V = np.asarray(study['V_reaction_2d'], dtype=float)
    # The separable survey reaction augments the base axis reaction.
    vx = V.mean(axis=1) + study.get('reaction_x', 0.0)
    vy = V.mean(axis=0) - V.mean() + study.get('reaction_y', 0.0)
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'], vx)
    ay = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'], vy)
    A = sparse.kronsum(ay, ax, format='csc')
    I = sparse.eye(nx * ny, format='csc')
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] * ny + ij[:, 1]
    rhs = modes.reshape(m, -1).T
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(A + float(mu) * I).solve(rhs)
        blocks.append(fields[sensors])
    return np.concatenate(blocks), fields.T.reshape(m, nx, ny)


def _screen_sparse(raw, y, sparse_banks, origin, design):
    # Infer the sparse wing's inverse gains and offsets without its spot checks,
    # then reject only checks inconsistent with that independent estimate.
    columns = []
    for mask, _, _ in sparse_banks:
        columns.extend((np.where(mask, raw, 0.0), mask.astype(float)))
    X = np.column_stack(columns)
    U, sv, _ = np.linalg.svd(design, full_matrices=False)
    U = U[:, sv > sv[0] * 1e-12]
    PX = X - U @ (U.T @ X)
    rhs = y - origin
    rhs -= U @ (U.T @ rhs)
    pinv = np.linalg.pinv(PX, rcond=1e-12)
    if not np.allclose(pinv @ PX, np.eye(X.shape[1]), atol=1e-7):
        return sparse_banks
    coef = -pinv @ rhs
    sigma = np.linalg.norm(rhs + PX @ coef) / np.sqrt(max(1, len(y) - U.shape[1] - X.shape[1]))
    screened = []
    for i, (mask, ref, readout) in enumerate(sparse_banks):
        C = np.column_stack((readout, np.ones(len(readout))))
        sl = slice(2*i, 2*i+2)
        tol = np.maximum(2e-7, 6 * sigma * np.linalg.norm(C @ pinv[sl], axis=1))
        keep = np.abs(C @ coef[sl] - ref) <= tol
        screened.append((mask, ref[keep], readout[keep]))
    return screened


def _calibrate(study, S, Z):
    rows = np.asarray(study['row_bank_id'], dtype=int)
    raw = np.asarray(study['y_obs_raw'], dtype=float).ravel()
    sweep = study['sensor_calibration_sweeps']
    if not isinstance(sweep, dict):
        sweep = {key: [row[key] for row in sweep] for key in ('bank_id', 'ref_signal', 'raw_readout')}
    banks = np.asarray(sweep['bank_id']).ravel()
    refs = np.asarray(sweep['ref_signal'], dtype=float).ravel()
    readouts = np.asarray(sweep['raw_readout'], dtype=float).ravel()
    y = np.zeros_like(raw)
    alternatives, free = [], []
    pending, reliable = {}, set()
    for b in range(4):
        mask = (banks == b) & np.isfinite(refs) & np.isfinite(readouts)
        x, r = refs[mask], readouts[mask]
        rowmask = rows == b
        if not rowmask.any():
            continue
        if x.size <= 2:
            pending[b] = (rowmask, x, r)
            continue
        i, j = np.triu_indices(x.size, 1)
        span = x[j] - x[i]
        pairs = (np.abs(span) > 1e-7) & ((r[j] - r[i]) * span > 0)
        if not pairs.any():
            if x.size:
                ref = np.median(x)
                pivot = np.median(r)
                y[rowmask] = ref
                free.append(np.where(rowmask, raw - pivot, 0.0))
            else:
                free.extend((np.where(rowmask, raw, 0.0), rowmask.astype(float)))
            continue
        gains = (r[j[pairs]] - r[i[pairs]]) / span[pairs]
        offsets = r[i[pairs]] - gains * x[i[pairs]]
        errors = np.abs(r[:, None] - x[:, None] * gains - offsets)
        inliers = errors <= 2e-7
        counts = inliers.sum(axis=0)
        max_count = counts.max()
        if max_count >= max(3, x.size // 2 + 1):
            idx = np.argmax(np.where(counts == max_count, np.abs(span[pairs]), -1.0))
            good = inliers[:, idx]
            gain, offset = np.linalg.lstsq(np.column_stack((x[good], np.ones(good.sum()))), r[good], rcond=None)[0]
            y[rowmask] = (raw[rowmask] - offset) / gain
            reliable.add(b)
        else:
            keep = counts == max_count
            alternatives.append((rowmask, (raw[rowmask, None] - offsets[keep]) / gains[keep]))
    origin = S.mean(axis=1) * float(study['total_budget'])
    budget_design = np.column_stack((S[:, 1:] - S[:, :1], Z))
    for wing in ((0, 1), (2, 3)):
        missing = [b for b in wing if b in pending]
        opposite = {0, 1, 2, 3}.difference(wing)
        if (missing and not alternatives and not free and opposite <= reliable
                and all(b in reliable or b in pending for b in wing)):
            for b, check in zip(missing, _screen_sparse(raw, y, [pending[b] for b in missing], origin, budget_design)):
                pending[b] = check
    for mask, x, r in pending.values():
        if x.size == 2 and abs(x[1] - x[0]) > 1e-7 and (x[1] - x[0]) * (r[1] - r[0]) > 0:
            inv_gain = (x[1] - x[0]) / (r[1] - r[0])
            y[mask] = x[0] + inv_gain * (raw[mask] - r[0])
        elif x.size:
            y[mask] = np.median(x)
            free.append(np.where(mask, raw - np.median(r), 0.0))
        else:
            free.extend((np.where(mask, raw, 0.0), mask.astype(float)))
    if not alternatives and not free:
        return y

    # Only use the source budget to anchor coefficients the sweeps cannot
    # identify. A budget-feasible signal is origin plus a zero-sum contrast.
    if free:
        design = budget_design
    else:
        origin = 0.0
        design = np.column_stack((S, Z))
    U, sv, _ = np.linalg.svd(design, full_matrices=False)
    U = U[:, sv > sv[0] * 1e-12]

    def project(v):
        return v - U @ (U.T @ v)

    X = np.column_stack(free) if free else None
    PX = project(X) if free else None
    best_score = np.inf
    for choices in product(*(range(values.shape[1]) for _, values in alternatives)):
        pred = y.copy()
        for (mask, values), i in zip(alternatives, choices):
            pred[mask] = values[:, i]
        residual = project(pred - origin)
        if free:
            inv_affine = np.linalg.lstsq(PX, -residual, rcond=None)[0]
            pred += X @ inv_affine
            residual += PX @ inv_affine
        score = residual @ residual
        if score < best_score:
            best_score, best = score, pred
    return best


def _profile(Z, data, ridge):
    if not Z.shape[1]:
        return data
    if ridge == 0:
        return data - Z @ np.linalg.lstsq(Z, data, rcond=None)[0]
    gram = Z.T @ Z + ridge * np.eye(Z.shape[1])
    return data - Z @ np.linalg.solve(gram, Z.T @ data)


def _box_simplex(H, c, upper, budget):
    n = c.size
    w = np.full(n, budget / n)
    if budget <= 0:
        return np.zeros(n), float(c.min())
    if budget >= n * upper:
        w.fill(upper)
        return w, float((H @ w + c).max())
    active = np.zeros(n, dtype=int)
    for _ in range(10 * n**2):
        free = np.flatnonzero(active == 0)
        g = H @ w + c
        if not free.size:
            lo = np.where(active == -1)[0]
            hi = np.where(active == 1)[0]
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            free = np.array([i, j])
        k = len(free)
        K = np.empty((k + 1, k + 1))
        K[:-1, :-1] = H[np.ix_(free, free)]
        K[:-1, -1] = -1.0
        K[-1, :-1] = 1.0
        K[-1, -1] = 0.0
        rhs = np.r_[-g[free], budget - w.sum()]
        try:
            result = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            result = np.linalg.lstsq(K, rhs, rcond=None)[0]
        d, nu = result[:-1], result[-1]
        limits = np.full(k, np.inf)
        down, up = d < -1e-14, d > 1e-14
        limits[down] = -w[free[down]] / d[down]
        limits[up] = (upper - w[free[up]]) / d[up]
        hit = np.argmin(limits)
        if limits[hit] < 1 - 1e-12:
            w[free] += max(0.0, limits[hit]) * d
            i = free[hit]
            active[i] = 1 if up[hit] else -1
            w[i] = upper if up[hit] else 0.0
            continue
        w[free] = np.clip(w[free] + d, 0, upper)
        bad = active * (H @ w + c - nu)
        i = np.argmax(bad)
        if bad[i] <= 1e-12:
            return w, float(nu)
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_w01_s00(study: dict) -> dict:
    S, terminal_modes = _transport(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrate(study, S, Z)
    data = _profile(Z, np.column_stack((S, y)), float(study['nuisance_ridge']))
    S_tilde, r_y = data[:, :-1], data[:, -1]
    ridge, upper, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = S_tilde.T @ S_tilde + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S_tilde.T @ r_y), upper, budget)
    residual = S_tilde @ w - r_y
    p = 0.5 * float(residual @ residual + ridge * (w @ w))
    g = S_tilde.T @ residual + ridge * w
    bound = p - g @ w + nu * budget + upper * np.minimum(0.0, g - nu).sum()
    cert = {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': np.maximum(g - nu, 0),
            'lam_upper': np.maximum(nu - g, 0), 'primal_objective': p,
            'dual_lower_bound': float(min(p, bound))}
    terminal = np.einsum('m,mij->ij', w, terminal_modes)
    return {'weights': w, 'terminal_field': terminal, 'kkt_certificate': cert}
