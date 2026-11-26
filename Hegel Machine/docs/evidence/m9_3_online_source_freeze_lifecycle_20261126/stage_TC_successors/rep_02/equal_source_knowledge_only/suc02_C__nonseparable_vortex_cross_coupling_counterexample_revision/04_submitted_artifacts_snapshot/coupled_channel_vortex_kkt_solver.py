"""Forward transport and certified box-simplex inversion for two tracer channels."""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _transport(length, diffusivity, velocity, robin):
    """Cell-centered harmonic diffusion, upwind advection, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float) / length**2
    v = np.asarray(velocity, dtype=float) / length
    r = robin / length
    h = 1.0 / k.size
    face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    diagonal = abs(v) / h
    diagonal[:-1] += face
    diagonal[1:] += face
    for i in (0, k.size - 1):
        diagonal[i] += r * k[i] / (h * (0.5 * r * h + k[i]))
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face - np.maximum(-v[:-1], 0) / h
    return diags((lower, diagonal, upper), (-1, 0, 1), shape=(k.size, k.size), format='csc')


def _spectral_basis(operator):
    # A = D Q diag(eigenvalues) Q.T D^-1, with positive diagonal D.
    lower = operator.diagonal(-1)
    upper = operator.diagonal(1)
    log_scale = np.r_[0., 0.5 * np.cumsum(np.log(lower / upper))]
    scale = np.exp(log_scale - 0.5 * (log_scale.max() + log_scale.min()))
    values, vectors = eigh_tridiagonal(operator.diagonal(), -np.sqrt(lower * upper))
    return values, scale[:, None] * vectors, vectors.T / scale


def _forward(ax, by, modes, shifts, pocket, sensors):
    m, nx, ny = modes.shape
    i, j = sensors.T
    ex, x_forward, x_inverse = _spectral_basis(ax)
    ey, y_forward, y_inverse = _spectral_basis(by)
    rhs_a = x_inverse @ modes @ y_inverse.T
    spectrum = ex[:, None] + ey[None, :]
    rhs_b = modes.reshape(m, nx * ny).T
    base_b = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    base_b += diags(pocket.ravel(), format='csc')
    identity = eye(nx * ny, format='csc')
    rows_a, rows_b = [], []
    for mu in shifts:
        terminal_a = x_forward @ (rhs_a / (spectrum + mu)) @ y_forward.T
        terminal_b = splu(base_b + mu * identity).solve(rhs_b).T.reshape(m, nx, ny)
        rows_a.append(terminal_a[:, i, j].T)
        rows_b.append(terminal_b[:, i, j].T)
    return np.vstack(rows_a + rows_b), terminal_a, terminal_b


def _profile(design, observations, nuisance, ridge):
    """Apply the same ridge residual maker to the design and observations."""
    data = np.column_stack((design, observations))
    if nuisance.shape[1]:
        q, sigma, _ = np.linalg.svd(nuisance, full_matrices=False)
        if ridge:
            shrink = sigma**2 / (sigma**2 + ridge)
        else:
            shrink = sigma > np.finfo(float).eps * max(nuisance.shape) * sigma[0]
        data -= q @ (shrink[:, None] * (q.T @ data))
    return data[:, :-1], data[:, -1]


def _multipliers(gradient, cap, budget):
    if budget == 0:
        nu = float(gradient.min())
    elif budget == cap * gradient.size:
        nu = float(gradient.max())
    else:
        rank = min(gradient.size - 1, max(0, int(np.ceil(budget / cap)) - 1))
        nu = float(np.partition(gradient, rank)[rank])
    return nu, np.maximum(gradient - nu, 0), np.maximum(nu - gradient, 0)


def _box_simplex(design, target, ridge, cap, budget):
    """Minimize on successive faces, using the global tangent gap to stop."""
    n = design.shape[1]
    a = np.vstack((design, np.sqrt(ridge) * np.eye(n)))
    y = np.r_[target, np.zeros(n)]
    if budget == 0 or budget == n * cap:
        return np.full(n, 0. if budget == 0 else cap)

    w = np.full(n, budget / n)
    bounds = np.zeros(n, dtype=int)  # -1: lower; 0: free; +1: upper
    for _ in range(40 * n):
        residual = a @ w - y
        gradient = a.T @ residual
        _, lower, upper = _multipliers(gradient, cap, budget)
        if lower @ w + upper @ (cap - w) <= 1e-11:
            return w

        free = np.flatnonzero(bounds == 0)
        if free.size >= 2:
            # Use one free coordinate to enforce the budget on this face.
            pivot = free[-1]
            rest = free[:-1]
            dz = np.linalg.lstsq(a[:, rest] - a[:, [pivot]], -residual, rcond=None)[0]
            direction = np.r_[dz, -dz.sum()]
            trial = w[free] + direction
            violated = (trial < 0) | (trial > cap)
            if np.any(violated):
                hits = np.flatnonzero(violated)
                values = np.where(direction[hits] > 0, cap, 0.)
                steps = (values - w[free[hits]]) / direction[hits]
                first = np.argmin(steps)
                w[free] = np.clip(w[free] + steps[first] * direction, 0., cap)
                index = free[hits[first]]
                w[index] = values[first]
                bounds[index] = 1 if values[first] == cap else -1
                continue
            w[free] = trial
            gradient = a.T @ (a @ w - y)
            _, lower, upper = _multipliers(gradient, cap, budget)
            if lower @ w + upper @ (cap - w) <= 1e-11:
                return w

        # Release the active bounds involved in the best feasible transfer.
        receivers = np.flatnonzero(bounds != 1)
        donors = np.flatnonzero(bounds != -1)
        receive = receivers[np.argmin(gradient[receivers])]
        donate = donors[np.argmax(gradient[donors])]
        bounds[[receive, donate]] = 0
    raise RuntimeError('Box-simplex solve did not reach the target certificate gap')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Return the source mixture, terminal tracer fields, and a KKT certificate."""
    ax = _transport(study['Lx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    by = _transport(study['Ly'], study['kappa_y'], study['vel_y'], study['robin_y'])
    raw, terminal_a, terminal_b = _forward(
        ax, by, np.asarray(study['source_modes'], dtype=float), study['mu_shifts'],
        np.asarray(study['V_pocket_channel_b'], dtype=float), np.asarray(study['sensor_ij']),
    )
    design, target = _profile(raw, np.asarray(study['y_obs']).ravel(),
                              np.asarray(study['Z_nuisance']), study['nuisance_ridge'])
    ridge, cap, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _box_simplex(design, target, ridge, cap, budget)
    residual = design @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    gradient = design.T @ residual + ridge * w
    nu, lower, upper = _multipliers(gradient, cap, budget)
    gap = float(lower @ w + upper @ (cap - w) + nu * (w.sum() - budget))
    certificate = {
        'weights': w.tolist(), 'nu_sum': nu,
        'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
        'primal_objective': objective,
        'dual_lower_bound': objective - max(0., gap),
    }
    return {
        'weights': w,
        'terminal_field_channel_a': np.einsum('m,mij->ij', w, terminal_a),
        'terminal_field_channel_b': np.einsum('m,mij->ij', w, terminal_b),
        'kkt_certificate': certificate,
    }
