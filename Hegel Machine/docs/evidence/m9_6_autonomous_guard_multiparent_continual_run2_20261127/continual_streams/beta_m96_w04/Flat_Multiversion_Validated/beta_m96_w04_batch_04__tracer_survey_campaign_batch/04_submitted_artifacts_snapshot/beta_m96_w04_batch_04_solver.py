"""Finite-volume forward solves and certified box-simplex inversion."""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import solve_sylvester
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


_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep_groups(sweep):
    banks = np.asarray(sweep['bank_id'])
    x = np.asarray(sweep['ref_signal'], dtype=float)
    z = np.asarray(sweep['raw_readout'], dtype=float)
    return [(x[banks == b], z[banks == b]) for b in range(4)]


def _line(x, z):
    if np.unique(x).size < 2:
        return None, 0
    i, j = np.triu_indices(len(x), 1)
    distinct = x[j] != x[i]
    i, j = i[distinct], j[distinct]
    gain = np.median((z[j] - z[i]) / (x[j] - x[i]))
    offset = np.median(z - gain * x)
    # Use the affine sweep precision, not the size of a potential EMI spike.
    tol = 2e-6 * max(1., np.median(abs(z)))
    keep = abs(z - gain * x - offset) <= tol
    if np.unique(x[keep]).size >= 2:
        fit = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))),
                              z[keep], rcond=None)[0]
        return fit, int(keep.sum())
    return None, 0


def _calibrate(study):
    groups = _sweep_groups(study['sensor_calibration_sweeps'])
    fits = [_line(x, z) for x, z in groups]
    rows = np.asarray(study['row_bank_id'])
    used = np.unique(rows)
    uncertain = [b for b in used if fits[b][1] < max(3, len(groups[b][0]) // 2 + 1)]
    if uncertain and _CALIBRATION_HISTORY.is_file():
        # A sweep is a snapshot: do not average different gain regimes across time.
        archive = json.loads(_CALIBRATION_HISTORY.read_text())
        prior = np.asarray([[_line(x, z)[0] for x, z in _sweep_groups(sweep)]
                            for sweep in archive], dtype=float)
        support = np.empty((len(prior), 4), dtype=int)
        for b, (x, z) in enumerate(groups):
            err = np.abs(prior[:, b, 0, None] * x + prior[:, b, 1, None] - z)
            support[:, b] = (err <= 3e-6 * np.maximum(1., abs(z))).sum(axis=1)
        # Other banks identify a recurring regime when this bank has no usable sweep.
        # Count banks, not samples, so a heavily sampled bank cannot dominate.
        votes = (support >= 2).sum(axis=1)
        for b in uncertain:
            best = max(range(len(prior)), key=lambda i: (support[i, b], votes[i], i))
            if fits[b][0] is None or support[best, b] >= fits[b][1]:
                fits[b] = (prior[best, b], support[best, b])
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    for b in used:
        fit, _ = fits[b]
        if fit is None or fit[0] <= 0:
            raise ValueError(f'Bank {b} has no identifiable positive calibration gain')
        sel = rows == b
        y[sel] = (y[sel] - fit[1]) / fit[0]
    return y


def _forward(study):
    nx, ny = study['nx'], study['ny']
    rx, ry = (np.asarray(study['reaction_' + a]) for a in 'xy')
    V = np.asarray(study['V_reaction_2d'])
    vx = V.mean(axis=1)
    vy = V.mean(axis=0) - V.mean()
    cross = V - vx[:, None] - vy[None, :]
    relative_cross = np.linalg.norm(cross) / max(np.linalg.norm(V), np.finfo(float).tiny)
    vortex = np.any(np.asarray(study['past_vortex_circulation_trace'])
                    > study['critical_vortex_circulation'])
    laminar = relative_cross <= 1e-6 and not vortex
    scale = 1.45 if vortex or relative_cross > 1e-4 else 1.0
    # The quiescent archive can carry a 2D copy of the base axis losses.
    # A distinct separable map is additional, just as in the full stencil.
    if laminar and not np.allclose(V, rx[:, None] + ry[None, :], rtol=1e-12, atol=1e-12):
        rx, ry = rx + vx, ry + vy
    X, Y = (
        _build_1d(study['L' + a], n, scale * np.asarray(study['kappa_' + a]),
                  study['vel_' + a], study['robin_' + a], r)
        for a, n, r in (('x', nx, rx), ('y', ny, ry))
    )
    size = nx * ny
    sources = np.asarray(study['source_modes'])
    ij = np.asarray(study['sensor_ij'])
    idx = ij[:, 0] * ny + ij[:, 1]
    samples = []
    if laminar:
        X, Yt = X.toarray(), Y.toarray().T
        ix = np.eye(nx)
        for mu in study['mu_shifts']:
            A = X + mu * ix
            U = np.column_stack([solve_sylvester(A, Yt, f).ravel() for f in sources])
            samples.append(U[idx])
    else:
        base = (sparse.kron(X, sparse.eye(ny), format='csc')
                + sparse.kron(sparse.eye(nx), Y, format='csc')
                + sparse.diags(V.ravel(), format='csc'))
        ident = sparse.eye(size, format='csc')
        rhs = sources.reshape(-1, size).T
        for mu in study['mu_shifts']:
            U = splu(base + mu * ident).solve(rhs)
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


def solve_beta_m96_w04_batch_04(study: dict) -> dict:
    y = _calibrate(study)
    S, U_terminal = _forward(study)
    S_tilde, r_y = _profile(study, S, y)
    w, cert = _box_simplex(S_tilde, r_y, study['source_ridge'],
                           study['u_box'], study['total_budget'])
    field = (U_terminal @ w).reshape(study['nx'], study['ny'])
    return dict(weights=w, terminal_field=field, kkt_certificate=cert)
