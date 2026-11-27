"""Calibrated source tomography for the laminar reaction-transport regime."""

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _build_1d(L, n, kappa, vel, robin, diag_add):
    """Return the lower, main, and upper diagonals of the FVM operator."""
    h = L / n
    kappa = np.asarray(kappa, dtype=float)
    speed = np.asarray(vel, dtype=float) / h
    diffusion = 2 * kappa[:-1] * kappa[1:] / ((kappa[:-1] + kappa[1:]) * h**2)
    main = np.asarray(diag_add, dtype=float) + np.abs(speed)
    main[:-1] += diffusion
    main[1:] += diffusion
    edge = np.array([0, n - 1])
    np.add.at(main, edge, robin * kappa[edge] / (h * (robin * h / 2 + kappa[edge])))
    # A missing upstream neighbor has zero inflow value, not a zero derivative.
    lower = -diffusion - np.maximum(speed[1:], 0)
    upper = -diffusion + np.minimum(speed[:-1], 0)
    return lower, main, upper


def _axis_basis(study, axis, reaction):
    lower, main, upper = _build_1d(
        study['L' + axis], len(reaction), study['kappa_' + axis],
        study['vel_' + axis], study['robin_' + axis], reaction,
    )
    # D^-1 A D is symmetric, so the same stable eigensystem handles every shift.
    scale = np.exp(np.r_[0., np.cumsum(0.5 * np.log(lower / upper))])
    values, basis = eigh_tridiagonal(main, -np.sqrt(lower * upper))
    return values, scale[:, None] * basis, basis.T / scale


def _response(study):
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    if (not np.allclose(v, vx[:, None] + vy[None, :], rtol=0, atol=1e-10)
            or np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('The study is outside the laminar separable regime.')
    dx, px, ix = _axis_basis(study, 'x', vx + study['reaction_x'])
    dy, py, iy = _axis_basis(study, 'y', vy + study['reaction_y'])
    modes = np.asarray(study['source_modes'], dtype=float)
    spectrum = ix @ modes @ iy.T
    eigsum = dx[:, None] + dy[None, :]
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = px @ (spectrum / (eigsum + shift)) @ py.T
        blocks.append(fields[:, i, j].T)
    return np.concatenate(blocks, axis=0), fields


def _fit_bank(x, y):
    """Fit an affine sweep, screening isolated readout spikes before refitting."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    good = np.abs(dx) > 1e-12 * max(1., np.ptp(x))
    if not good.any():
        raise ValueError('A bank sweep needs at least two distinct references.')
    gain = np.median((y[j[good]] - y[i[good]]) / dx[good])
    offset = np.median(y - gain * x)
    residual = y - gain * x - offset
    mad = np.median(np.abs(residual))
    keep = np.abs(residual) <= max(5 * 1.4826 * mad, 1e-12 * max(1., np.max(np.abs(y))))
    if not keep.any() or np.ptp(x[keep]) == 0:
        return gain, offset
    x, y = x[keep], y[keep]
    centered = x - x.mean()
    gain = (centered @ (y - y.mean())) / (centered @ centered)
    return gain, y.mean() - gain * x.mean()


def _calibrated_observations(study):
    sweep = study['sensor_calibration_sweeps']
    banks = np.asarray(sweep['bank_id'])
    ref = np.asarray(sweep['ref_signal'], dtype=float)
    raw = np.asarray(sweep['raw_readout'], dtype=float)
    row_bank = np.asarray(study['row_bank_id'])
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    for b in np.unique(row_bank):
        choose = banks == b
        gain, offset = _fit_bank(ref[choose], raw[choose])
        rows = row_bank == b
        y[rows] = (y[rows] - offset) / gain
    return y


def _profile(study, design, y):
    z = np.asarray(study['Z_nuisance'], dtype=float)
    ridge = float(study['nuisance_ridge'])
    values = np.column_stack((design, y))
    if ridge == 0:
        beta = np.linalg.lstsq(z, values, rcond=None)[0]
    else:
        beta = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ values)
    residual = values - z @ beta
    return residual[:, :-1], residual[:, -1]


def _box_simplex(design, target, ridge, upper, budget):
    n = design.shape[1]
    h = design.T @ design + ridge * np.eye(n)
    c = design.T @ target
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper.
    if budget == 0:
        state[:] = -1
    elif budget == n * upper:
        state[:] = 1
    tol = 1e-12
    for _ in range(20 * n * n):
        g = h @ w - c
        free = np.flatnonzero(state == 0)
        if not len(free):
            donors, receivers = np.flatnonzero(state == 1), np.flatnonzero(state == -1)
            if not len(donors) or not len(receivers):
                nu = g.max() if len(donors) else g.min()
                break
            donor, receiver = donors[np.argmax(g[donors])], receivers[np.argmin(g[receivers])]
            if g[donor] <= g[receiver] + tol:
                nu = (g[donor] + g[receiver]) / 2
                break
            state[[donor, receiver]] = 0
            continue
        k = len(free)
        system = np.ones((k + 1, k + 1))
        system[:k, :k] = h[np.ix_(free, free)]
        system[-1, -1] = 0
        step_nu = np.linalg.solve(system, np.r_[-g[free], budget - w.sum()])
        step, nu = step_nu[:-1], -step_nu[-1]
        candidate = w[free] + step
        crosses = (candidate < -tol) | (candidate > upper + tol)
        if np.any(crosses):
            q = np.flatnonzero(crosses)
            caps = np.where(step[q] > 0, upper, 0.)
            fractions = (caps - w[free[q]]) / step[q]
            j = np.argmin(fractions)
            w[free] += fractions[j] * step
            idx = free[q[j]]
            w[idx] = caps[j]
            state[idx] = 1 if step[q[j]] > 0 else -1
            continue
        w[free] = np.clip(candidate, 0, upper)
        g = h @ w - c
        violation = state * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            break
        state[j] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge.')
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    lower_bound = objective - g @ w + nu * budget + upper * np.minimum(g - nu, 0).sum()
    return w, {'weights': w.copy(), 'nu_sum': float(nu),
               'lam_lower': np.maximum(g - nu, 0), 'lam_upper': np.maximum(nu - g, 0),
               'primal_objective': float(objective),
               'dual_lower_bound': float(min(objective, lower_bound))}


def solve_beta_m96_w04_s00(study: dict) -> dict:
    design, fields = _response(study)
    design, target = _profile(study, design, _calibrated_observations(study))
    w, cert = _box_simplex(design, target, float(study['source_ridge']),
                           float(study['u_box']), float(study['total_budget']))
    return {'weights': w, 'terminal_field': np.einsum('m,mij->ij', w, fields),
            'kkt_certificate': cert}
