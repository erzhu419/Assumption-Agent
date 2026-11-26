"""Dual-channel tracer inversion with a box-simplex optimality certificate."""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_stencil(length, diffusivity, velocity, robin):
    """Cell-centered harmonic diffusion, upwind advection, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    upstream = np.maximum(v, 0) / h
    downstream = np.maximum(-v, 0) / h
    lower = -face - upstream[1:]
    upper = -face - downstream[:-1]
    main = upstream + downstream
    main[1:] += face
    main[:-1] += face
    ends = [0, k.size - 1]
    main[ends] += (robin / h) / (1 + robin * h / (2 * k[ends]))
    return lower, main, upper


def _axis_spectrum(stencil):
    lower, main, upper = stencil
    # D^{-1} A D is symmetric. 'left' and 'right' are inverse transforms.
    d = np.r_[1.0, np.cumprod(np.sqrt(lower / upper))]
    eigenvalues, q = eigh_tridiagonal(main, -np.sqrt(lower * upper))
    return eigenvalues, q.T / d, d[:, None] * q


def _responses(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    i, j = np.asarray(study['sensor_ij']).T
    sx, sy = (_axis_stencil(study['L' + axis], study['kappa_' + axis],
                           study['vel_' + axis], study['robin_' + axis])
              for axis in ('x', 'y'))
    ex, lx, rx = _axis_spectrum(sx)
    ey, ly, ry = _axis_spectrum(sy)
    spectral_sources = lx @ sources @ ly.T
    eigenvalue_sum = ex[:, None] + ey

    ax = diags(sx, [-1, 0, 1], format='csc')
    by = diags(sy, [-1, 0, 1], format='csc')
    base = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    pocket = np.asarray(study['V_pocket_channel_b'], dtype=float).ravel()
    rhs = sources.reshape(m, nx * ny).T
    a_rows, b_rows = [], []
    for shift in study['mu_shifts']:
        a = rx @ (spectral_sources / (eigenvalue_sum + shift)) @ ry.T
        factor = splu(base + diags(pocket + shift, format='csc'))
        b = factor.solve(rhs).T.reshape(m, nx, ny)
        a_rows.append(a[:, i, j].T)
        b_rows.append(b[:, i, j].T)
    return np.vstack(a_rows + b_rows), a, b


def _profile(design, target, nuisance, ridge):
    # The survey objective uses ordinary ridge-fit residuals, not the
    # square-root Schur-complement transform of a joint ridge objective.
    u, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    if ridge == 0:
        cutoff = np.finfo(float).eps * max(nuisance.shape) * singular.max(initial=0)
        shrink = singular > cutoff
    else:
        shrink = singular**2 / (singular**2 + ridge)
    data = np.column_stack((design, target))
    residual = data - u @ (shrink[:, None] * (u.T @ data))
    return residual[:, :-1], residual[:, -1]


def _box_simplex_qp(design, target, ridge, cap, budget):
    """Optimize on equality-constrained faces, adding/releasing box bounds."""
    n = design.shape[1]
    if budget == 0:
        return np.zeros(n)
    if budget == n * cap:
        return np.full(n, cap)

    h = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1: lower, +1: upper
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if free.size == 0:
            g = h @ w - linear
            upper = np.flatnonzero(active == 1)
            lower = np.flatnonzero(active == -1)
            i, j = upper[np.argmax(g[upper])], lower[np.argmin(g[lower])]
            if g[i] <= g[j] + 1e-12:
                return w
            active[[i, j]] = 0
            continue

        k = free.size
        system = np.empty((k + 1, k + 1))
        system[:k, :k] = h[np.ix_(free, free)]
        system[:k, k] = system[k, :k] = 1.0
        system[k, k] = 0.0
        rhs = np.r_[linear[free] - h[np.ix_(free, fixed)] @ w[fixed],
                    budget - w[fixed].sum()]
        solution = np.linalg.solve(system, rhs)
        trial, nu = solution[:-1], -solution[-1]
        current = w[free]
        below, above = trial < -1e-13, trial > cap + 1e-13
        if np.any(below | above):
            fraction = np.full(k, np.inf)
            fraction[below] = current[below] / (current[below] - trial[below])
            fraction[above] = (cap - current[above]) / (trial[above] - current[above])
            hit = np.argmin(fraction)
            w[free] = np.clip(current + fraction[hit] * (trial - current), 0, cap)
            active[free[hit]] = -1 if below[hit] else 1
            w[free[hit]] = 0.0 if below[hit] else cap
            continue

        w[free] = np.clip(trial, 0, cap)
        if fixed.size == 0:
            return w
        g = h @ w - linear
        wrong_sign = active[fixed] * (g[fixed] - nu)
        release = np.argmax(wrong_sign)
        if wrong_sign[release] <= 1e-12:
            return w
        active[fixed[release]] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, target, ridge, cap, budget, w):
    residual = design @ w - target
    objective = 0.5 * (float(residual @ residual) + ridge * float(w @ w))
    gradient = design.T @ residual + ridge * w
    # The linearized box-simplex problem fills coordinates in gradient order.
    # Its cutoff gradient maximizes the tangential lower bound over nu.
    cutoff = 0 if cap == 0 else int(np.ceil(budget / cap)) - 1
    nu = float(np.sort(gradient)[np.clip(cutoff, 0, w.size - 1)])
    reduced = gradient - nu
    lower = objective - float(gradient @ w) + nu * budget
    lower += cap * float(np.minimum(0.0, reduced).sum())
    return {'weights': w.copy(), 'nu_sum': nu,
            'lam_lower': np.maximum(reduced, 0.0),
            'lam_upper': np.maximum(-reduced, 0.0),
            'primal_objective': objective,
            'dual_lower_bound': min(objective, lower)}


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Return source weights, last-shift fields, and a KKT certificate."""
    design, a, b = _responses(study)
    design, target = _profile(design, np.asarray(study['y_obs']).ravel(),
                              study['Z_nuisance'], study['nuisance_ridge'])
    ridge, cap, budget = (float(study[key]) for key in
                          ('source_ridge', 'u_box', 'total_budget'))
    weights = _box_simplex_qp(design, target, ridge, cap, budget)
    return {'weights': weights,
            'terminal_field_channel_a': np.einsum('m,mij->ij', weights, a),
            'terminal_field_channel_b': np.einsum('m,mij->ij', weights, b),
            'kkt_certificate': _certificate(design, target, ridge, cap, budget, weights)}
