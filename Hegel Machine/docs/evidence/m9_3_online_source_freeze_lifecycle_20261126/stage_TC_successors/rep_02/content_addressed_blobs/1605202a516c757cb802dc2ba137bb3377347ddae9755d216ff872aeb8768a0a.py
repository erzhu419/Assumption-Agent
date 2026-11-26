"""Dual-channel tracer inversion and box-simplex optimality certificate."""

import numpy as np
from scipy import sparse
from scipy.linalg import helmert, schur
from scipy.linalg.lapack import dtrsyl
from scipy.sparse.linalg import splu


def _cell_transport(length, diffusivity, velocity, robin):
    """Harmonic face diffusion, row-wise upwind transport, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float) / length**2
    v = np.asarray(velocity, dtype=float) / length
    r = robin / length
    h = 1.0 / k.size
    face = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    main = np.abs(v) / h
    main[:-1] += face
    main[1:] += face
    for i in (0, k.size - 1):
        main[i] += r * k[i] / (h * (k[i] + 0.5 * r * h))
    return (np.diag(main)
            + np.diag(-face - np.maximum(v[1:], 0) / h, -1)
            + np.diag(-face - np.maximum(-v[:-1], 0) / h, 1))


def _responses(ax, by, shifts, modes, pocket, sensors):
    m, nx, ny = modes.shape
    # Schur bases preserve the separable Sylvester equation even for upwinding.
    tx, qx = schur(ax, output='real')
    ty, qy = schur(by, output='real')
    rhs_a = qx.T @ modes @ qy
    rhs_b = modes.reshape(m, -1).T
    base_b = (sparse.kron(ax, sparse.eye(ny), format='csc')
              + sparse.kron(sparse.eye(nx), by, format='csc')
              + sparse.diags(pocket.ravel(), format='csc'))
    eye_x = np.eye(nx)
    eye_b = sparse.eye(nx * ny, format='csc')
    i, j = sensors.T
    sa = np.empty((len(shifts), len(i), m))
    sb = np.empty_like(sa)
    for k, shift in enumerate(shifts):
        tx_shift = tx + shift * eye_x
        a = np.empty_like(modes)
        for p in range(m):
            field, scale, info = dtrsyl(tx_shift, ty, rhs_a[p], tranb='T')
            if info != 0 or scale == 0:
                raise np.linalg.LinAlgError('Singular or ill-conditioned Channel A Sylvester equation')
            a[p] = qx @ (field / scale) @ qy.T
        b = splu(base_b + shift * eye_b).solve(rhs_b).T.reshape(m, nx, ny)
        sa[k] = a[:, i, j].T
        sb[k] = b[:, i, j].T
    return np.concatenate((sa.reshape(-1, m), sb.reshape(-1, m))), a, b


def _ridge_residuals(design, data, ridge):
    """Apply I - Z (Z.T Z + ridge I)^(-1) Z.T to both S and y."""
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    if ridge == 0:
        if not singular.size:
            return data
        keep = singular > np.finfo(float).eps * max(design.shape) * singular[0]
        u = u[:, keep]
        return data - u @ (u.T @ data)
    shrink = singular**2 / (singular**2 + ridge)
    return data - u @ (shrink[:, None] * (u.T @ data))


def _box_simplex(design, target, cap, budget):
    """Minimize a least-squares objective on a capped simplex by active faces."""
    m = design.shape[1]
    w = np.full(m, budget / m)
    if budget == 0 or budget == m * cap:
        return w
    # Active coordinates are fixed at 0 (-1) or cap (+1).
    active = np.zeros(m, dtype=int)
    for _ in range(50 * m):
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active == -1)
        upper = np.flatnonzero(active == 1)
        residual = design @ w - target
        if free.size > 1:
            basis = helmert(free.size, full=False).T
            delta = basis @ np.linalg.lstsq(design[:, free] @ basis, -residual, rcond=None)[0]
            trial = w[free] + delta
            outside = (trial < 0) | (trial > cap)
            if np.any(outside):
                hit = np.flatnonzero(outside)
                bounds = np.where(trial[hit] < 0, 0.0, cap)
                steps = (bounds - w[free[hit]]) / delta[hit]
                first = np.argmin(steps)
                w[free] = np.clip(w[free] + steps[first] * delta, 0.0, cap)
                idx = free[hit[first]]
                w[idx] = bounds[first]
                active[idx] = -1 if bounds[first] == 0 else 1
                continue
            w[free] = trial
            residual = design @ w - target
        g = design.T @ residual
        lo = lower[np.argmin(g[lower])] if lower.size else None
        hi = upper[np.argmax(g[upper])] if upper.size else None
        # Pick the most violated bound; at a vertex a feasible move requires a pair.
        tol = 1e-12 / max(cap, 1.0)
        if not free.size:
            if g[hi] - g[lo] <= tol:
                return w
            active[[lo, hi]] = 0
        else:
            nu = np.mean(g[free])
            bad_lo = nu - g[lo] if lo is not None else 0.0
            bad_hi = g[hi] - nu if hi is not None else 0.0
            if max(bad_lo, bad_hi) <= tol:
                return w
            active[lo if bad_lo >= bad_hi else hi] = 0
    raise RuntimeError('Box-simplex active-set iteration did not converge')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Fit the shared source weights and return both terminal mixture fields."""
    ax, by = [_cell_transport(study['L' + d], study['kappa_' + d],
                              study['vel_' + d], study['robin_' + d]) for d in 'xy']
    design, a, b = _responses(ax, by, np.asarray(study['mu_shifts']),
                              np.asarray(study['source_modes'], dtype=float),
                              np.asarray(study['V_pocket_channel_b']),
                              np.asarray(study['sensor_ij'], dtype=int))
    data = _ridge_residuals(np.asarray(study['Z_nuisance']),
                            np.column_stack((design, study['y_obs'])), study['nuisance_ridge'])
    design, target = data[:, :-1], data[:, -1]
    cap, budget, ridge = (float(study[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    m = design.shape[1]
    w = _box_simplex(np.vstack((design, np.sqrt(ridge) * np.eye(m))),
                     np.r_[target, np.zeros(m)], cap, budget)
    residual = design @ w - target
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    g = design.T @ residual + ridge * w
    # A budget quantile of g makes the linearized box-simplex bound tightest.
    rank = int(np.clip(np.ceil(budget / cap) - 1, 0, m - 1)) if cap > 0 else 0
    nu = np.partition(g, rank)[rank]
    lower = np.maximum(g - nu, 0.0)
    upper = np.maximum(nu - g, 0.0)
    # This is J - L_KKT, evaluated as nonnegative complementarity terms.
    gap = lower @ w + upper @ (cap - w)
    certificate = dict(weights=w.copy(), nu_sum=float(nu), lam_lower=lower, lam_upper=upper,
                       primal_objective=float(primal),
                       dual_lower_bound=float(np.nextafter(primal - gap, -np.inf)))
    return dict(weights=w, terminal_field_channel_a=np.einsum('m,mij->ij', w, a),
                terminal_field_channel_b=np.einsum('m,mij->ij', w, b),
                kkt_certificate=certificate)
