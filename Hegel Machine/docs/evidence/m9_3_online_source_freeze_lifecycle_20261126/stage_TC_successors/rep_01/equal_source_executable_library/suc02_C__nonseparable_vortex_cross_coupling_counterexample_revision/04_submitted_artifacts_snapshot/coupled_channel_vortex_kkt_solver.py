"""Dual-channel tracer inversion with a box-simplex optimality certificate."""

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_operator(length, diffusion, velocity, robin):
    k = np.asarray(diffusion, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face + np.minimum(v[:-1], 0) / h
    center = np.zeros_like(k)
    center[1:] -= lower
    center[:-1] -= upper
    center[0] += (robin / (1 + robin * h / (2 * k[0])) + max(v[0], 0)) / h
    center[-1] += (robin / (1 + robin * h / (2 * k[-1])) - min(v[-1], 0)) / h
    return lower, center, upper


def _axis_spectrum(lower, center, upper):
    # D^-1 A D is symmetric: the diagonal scaling balances each face.
    log_scale = np.r_[0., np.cumsum(0.5 * np.log(lower / upper))]
    scale = np.exp(log_scale - np.mean(log_scale))
    eigenvalues, q = eigh_tridiagonal(center, -np.sqrt(lower * upper))
    return eigenvalues, scale[:, None] * q, q.T / scale[None, :]


def _forward(study):
    src = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = src.shape
    a = _axis_operator(study['Lx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    b = _axis_operator(study['Ly'], study['kappa_y'], study['vel_y'], study['robin_y'])
    ex, x, xi = _axis_spectrum(*a)
    ey, y, yi = _axis_spectrum(*b)
    coeff = xi @ src @ yi.T
    spectrum = ex[:, None] + ey[None, :]

    ax = diags(a, [-1, 0, 1], format='csc', shape=(nx, nx))
    by = diags(b, [-1, 0, 1], format='csc', shape=(ny, ny))
    spatial = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    potential = np.asarray(study['V_pocket_channel_b'], dtype=float).ravel()
    rhs = np.asfortranarray(src.reshape(m, nx * ny).T)
    i, j = np.asarray(study['sensor_ij']).T
    reads_a, reads_b = [], []
    for shift in study['mu_shifts']:
        ua = x @ (coeff / (spectrum + shift)) @ y.T
        factor = splu(spatial + diags(potential + shift, format='csc'))
        ub = factor.solve(rhs).T.reshape(m, nx, ny)
        reads_a.append(ua[:, i, j].T)
        reads_b.append(ub[:, i, j].T)
    return np.vstack(reads_a + reads_b), ua, ub


def _profile(design, target, nuisance, ridge):
    # The study defines the transformed data as ridge-fit residuals, not
    # the square root of the Schur complement.
    data = np.column_stack((design, target))
    if ridge > 0:
        gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
        coefficients = solve(gram, nuisance.T @ data, assume_a='pos')
    else:
        coefficients = np.linalg.lstsq(nuisance, data, rcond=None)[0]
    residuals = data - nuisance @ coefficients
    return residuals[:, :-1], residuals[:, -1]


def _minimize(design, target, ridge, cap, budget):
    n = design.shape[1]
    if budget == 0 or budget >= n * cap:
        return np.full(n, 0. if budget == 0 else cap)
    hess = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1: lower bound, 0: free, 1: upper bound
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        g = hess @ w - linear
        if free.size:
            # Minimize on the current face while preserving the total budget.
            k = free.size
            system = np.ones((k + 1, k + 1))
            system[:-1, :-1] = hess[np.ix_(free, free)]
            system[-1, -1] = 0.
            rhs = np.r_[-g[free], 0.]
            step = (solve(system, rhs, assume_a='sym') if ridge > 0 else
                    np.linalg.lstsq(system, rhs, rcond=None)[0])[:-1]
            step -= step.mean()
            current = w[free]
            endpoint = current + step
            hits = np.flatnonzero((endpoint < -1e-14) | (endpoint > cap + 1e-14))
            if hits.size:
                upper_hit = step[hits] > 0
                fractions = (np.where(upper_hit, cap, 0.) - current[hits]) / step[hits]
                first = np.argmin(fractions)
                w[free] = current + max(0., fractions[first]) * step
                bound = free[hits[first]]
                state[bound] = 1 if upper_hit[first] else -1
                w[bound] = cap if upper_hit[first] else 0.
                continue
            w[free] = np.clip(endpoint, 0., cap)
            g = hess @ w - linear
            nu = np.mean(g[free])
            violations = np.where(state < 0, nu - g, np.where(state > 0, g - nu, 0.))
            worst = np.argmax(violations)
            if violations[worst] <= 1e-12:
                return w
            state[worst] = 0
        else:
            # At a vertex, only transfers from an upper to a lower bound can help.
            low = np.flatnonzero(state < 0)
            high = np.flatnonzero(state > 0)
            receiver = low[np.argmin(g[low])]
            donor = high[np.argmax(g[high])]
            if g[donor] - g[receiver] <= 1e-12:
                return w
            state[[receiver, donor]] = 0
    raise RuntimeError('box-simplex active set did not converge')


def _certificate(design, target, w, ridge, cap, budget):
    residual = design @ w - target
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    g = design.T @ residual + ridge * w
    # The marginal sorted gradient solves the supporting linear problem.
    index = min(int(np.floor(budget / cap)), w.size - 1) if cap else 0
    nu = np.partition(g, index)[index]
    reduced = g - nu
    lower = np.maximum(reduced, 0.)
    upper = np.maximum(-reduced, 0.)
    gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - budget)
    return dict(weights=w, nu_sum=float(nu), lam_lower=lower, lam_upper=upper,
                primal_objective=float(objective),
                dual_lower_bound=float(objective - max(0., gap)))


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    design, ua, ub = _forward(study)
    design, target = _profile(design, study['y_obs'], study['Z_nuisance'], study['nuisance_ridge'])
    ridge, cap, budget = (study[key] for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _minimize(design, target, ridge, cap, budget)
    return dict(weights=w, terminal_field_channel_a=np.einsum('m,mij->ij', w, ua),
                terminal_field_channel_b=np.einsum('m,mij->ij', w, ub),
                kkt_certificate=_certificate(design, target, w, ridge, cap, budget))
