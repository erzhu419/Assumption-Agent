"""Finite-volume forward solves and certified box-simplex inversion."""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    kappa = np.asarray(kappa, dtype=float)
    vel = np.asarray(vel, dtype=float)
    face = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    main = np.broadcast_to(diag_add, (n,)).astype(float, copy=True) + abs(vel) / h
    main[:-1] += face
    main[1:] += face
    for i in (0, n - 1):
        main[i] += robin * kappa[i] / (h * (robin * h / 2 + kappa[i]))
    return sparse.diags(
        (-face - np.maximum(vel[1:], 0) / h,
         main,
         -face + np.minimum(vel[:-1], 0) / h),
        (-1, 0, 1), format='csc',
    )


def _sweep_columns(sweep):
    keys = ('bank_id', 'ref_signal', 'raw_readout')
    if isinstance(sweep, list):
        return tuple(np.asarray([row[key] for row in sweep]) for key in keys)
    return tuple(np.asarray(sweep[key]) for key in keys)


def _calibration_line(x, z, prior=None):
    """Fit an affine line by consensus, excluding isolated EMI readings."""
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (z[j] - z[i]) / (x[j] - x[i])
    positive = slopes > 0
    slopes, i = slopes[positive], i[positive]
    if not slopes.size:
        return prior
    offsets = z[i] - slopes * x[i]
    tol = 1.5e-6 * max(1., np.max(abs(z)))
    errors = abs(slopes[:, None] * x + offsets[:, None] - z)
    masks = errors <= tol
    counts = masks.sum(axis=1)
    best = np.flatnonzero(counts == counts.max())
    if prior is not None:
        pg, po = prior
        prior_count = np.count_nonzero(abs(pg * x + po - z) <= tol)
        # A slope supported by only its defining pair cannot adjudicate a
        # competing historical line. Use the trusted prefix in that case.
        if counts[best[0]] <= 2 and 0 < prior_count < len(x):
            return prior
        if len(best) > 1:
            distance = np.mean((slopes[best, None] * x + offsets[best, None]
                                - pg * x - po)**2, axis=1)
            best = best[distance == distance.min()]
    costs = np.sum(np.where(masks[best], errors[best]**2, 0.), axis=1)
    keep = masks[best[np.argmin(costs)]]
    A = np.column_stack((x[keep], np.ones(np.count_nonzero(keep))))
    return np.linalg.lstsq(A, z[keep], rcond=None)[0]


def _archived_calibration():
    path = '/tmp/hegel_continual_archive/archive_calibration_history.json'
    with open(path) as f:
        history = json.load(f)
    priors = {}
    for entry in history if isinstance(history, list) else [history]:
        banks, x, z = _sweep_columns(entry.get('sensor_calibration_sweeps', entry))
        for bank in np.unique(banks):
            sel = banks == bank
            line = _calibration_line(x[sel], z[sel])
            if line is not None:
                priors[bank] = line
    return priors


def _calibrate(study):
    banks, refs, raw = _sweep_columns(study['sensor_calibration_sweeps'])
    rows = np.asarray(study['row_bank_id'])
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    priors = None
    for bank in np.unique(rows):
        sel = banks == bank
        x, z = refs[sel], raw[sel]
        line = _calibration_line(x, z)
        if (line is None or np.unique(x).size < 3
                or np.count_nonzero(abs(line[0] * x + line[1] - z) <=
                                    1.5e-6 * max(1., np.max(abs(z)))) <= 2):
            if priors is None:
                priors = _archived_calibration()
            line = _calibration_line(x, z, priors.get(bank))
        if line is None:
            raise ValueError(f'Bank {bank} has no identifiable calibration line')
        gain, offset = line
        sel = rows == bank
        y[sel] = (y[sel] - offset) / gain
    return y


def _forward(study):
    nx, ny = study['nx'], study['ny']
    rx, ry = (np.asarray(study['reaction_' + a]) for a in 'xy')
    V = np.asarray(study['V_reaction_2d'])
    # The axis arrays are the base losses. A distinct map is an additional loss.
    # If the map is the base loss itself in a quiescent study, count it once.
    axes_only = (
        np.max(study['past_vortex_circulation_trace']) <= study['critical_vortex_circulation']
        and np.allclose(V, rx[:, None] + ry[None, :], rtol=1e-12, atol=1e-12)
    )
    X, Y = (
        _build_1d(study['L' + a], n, study['kappa_' + a], study['vel_' + a],
                  study['robin_' + a], r)
        for a, n, r in (('x', nx, rx), ('y', ny, ry))
    )
    size = nx * ny
    base = (sparse.kron(X, sparse.eye(ny), format='csc')
            + sparse.kron(sparse.eye(nx), Y, format='csc'))
    if not axes_only:
        base += sparse.diags(V.ravel(), format='csc')
    ident = sparse.eye(size, format='csc')
    src = np.asarray(study['source_modes']).reshape(-1, size).T
    ij = np.asarray(study['sensor_ij'])
    idx = ij[:, 0] * ny + ij[:, 1]
    samples = []
    for mu in study['mu_shifts']:
        U = splu(base + mu * ident).solve(src)
        samples.append(U[idx])
    return np.concatenate(samples, axis=0), U


def _profile(study, S, y):
    data = np.column_stack((S, y))
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    Q, d, _ = np.linalg.svd(Z, full_matrices=False)
    d2 = d * d
    denom = d2 + study['nuisance_ridge']
    if study['nuisance_ridge'] == 0 and d.size:
        denom = np.where(d > max(Z.shape) * np.finfo(float).eps * d[0], d2, np.inf)
    data -= Q @ ((d2 / denom)[:, None] * (Q.T @ data))
    return data[:, :-1], data[:, -1]


def _box_simplex(S, y, ridge, upper, budget):
    m = S.shape[1]
    u = np.broadcast_to(upper, (m,)).astype(float)
    H = S.T @ S + ridge * np.eye(m)
    c = S.T @ y
    if budget < 0 or budget > u.sum():
        raise ValueError('Infeasible box-simplex budget')
    tol = 2e-12
    lower = u == 0
    high = np.zeros(m, dtype=bool)
    w = np.zeros(m)
    nu = 0.
    if budget == 0:
        lower[:] = True
        nu = np.max(c)
    elif budget == u.sum():
        w = u.copy()
        high = ~lower
        nu = np.min(c - H @ w)
    else:
        w = u * (budget / u.sum())
        for _ in range(20 * m * m):
            free = ~(lower | high)
            if not np.any(free):
                grad = H @ w - c
                lo = np.flatnonzero(lower & (u > 0))
                hi = np.flatnonzero(high)
                i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
                if grad[i] >= grad[j] - tol:
                    nu = -0.5 * (grad[i] + grad[j])
                    break
                lower[i] = high[j] = False
                continue
            Hff = H[np.ix_(free, free)]
            rhs = c[free] - H[np.ix_(free, high)] @ u[high]
            ones = np.ones(free.sum())
            kkt = np.block([[Hff, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            step = np.linalg.solve(kkt, np.r_[rhs, budget - u[high].sum()])
            nu = step[-1]
            idx = np.flatnonzero(free)
            target = step[:-1]
            direction = target - w[idx]
            ratios = np.full(idx.size, np.inf)
            below = target < -tol
            above = target > u[idx] + tol
            ratios[below] = -w[idx[below]] / direction[below]
            ratios[above] = (u[idx[above]] - w[idx[above]]) / direction[above]
            hit = np.argmin(ratios)
            if ratios[hit] < 1:
                w[idx] += max(0., ratios[hit]) * direction
                j = idx[hit]
                if below[hit]:
                    lower[j], w[j] = True, 0.
                else:
                    high[j], w[j] = True, u[j]
                continue
            w[idx] = np.clip(target, 0., u[idx])
            grad = H @ w - c + nu
            violation = np.zeros(m)
            violation[lower & (u > 0)] = -grad[lower & (u > 0)]
            violation[high] = grad[high]
            j = np.argmax(violation)
            if violation[j] <= tol:
                break
            lower[j] = high[j] = False
        else:
            raise RuntimeError('Box-simplex active set failed to converge')
    grad = H @ w - c + nu
    lam_lower = np.where(lower, np.maximum(grad, 0.), 0.)
    lam_upper = np.where(high | (u == 0), np.maximum(-grad, 0.), 0.)
    residual = S @ w - y
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    # The public certificate defines nu_sum by g - nu_sum - lower + upper = 0.
    nu_sum = -nu
    g = S.T @ residual + ridge * w
    # Minimize the supporting hyperplane, relaxed by the sum multiplier,
    # over the box. This is the independently checked lower bound.
    dual = primal - g @ w + nu_sum * budget + u @ np.minimum(0., g - nu_sum)
    return w, dict(weights=w.tolist(), nu_sum=float(nu_sum), lam_lower=lam_lower.tolist(),
                   lam_upper=lam_upper.tolist(), primal_objective=float(primal),
                   dual_lower_bound=float(min(primal, dual)))


def solve_beta_m96_w04_batch_01(study: dict) -> dict:
    y = _calibrate(study)
    S, terminal = _forward(study)
    S_tilde, r_y = _profile(study, S, y)
    fits = [_box_simplex(S_tilde, r_y, study['source_ridge'], study['u_box'], budget)
            for budget in study['budget_grid']]
    weights = np.stack([w for w, _ in fits])
    field = (terminal @ weights[1]).reshape(study['nx'], study['ny'])
    return dict(weights_by_budget=weights, terminal_field_mid_budget=field,
                kkt_certificates=[cert for _, cert in fits])
