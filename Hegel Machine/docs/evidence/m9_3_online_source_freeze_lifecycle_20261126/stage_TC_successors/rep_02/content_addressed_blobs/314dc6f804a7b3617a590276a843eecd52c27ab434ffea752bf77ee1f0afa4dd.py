"""Separable shifted transport and tangential box-simplex certificates."""

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _axis_stencil(length, kappa, velocity, robin):
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = (2.0 * k[:-1] * k[1:]) / ((k[:-1] + k[1:]) * h**2)
    diagonal = np.abs(v) / h
    diagonal[:-1] += face
    diagonal[1:] += face
    ends = np.array([0, k.size - 1])
    np.add.at(diagonal, ends, robin * k[ends] / (h * (k[ends] + 0.5 * h * robin)))
    lower = -face - np.maximum(v[1:], 0.0) / h
    upper = -face - np.maximum(-v[:-1], 0.0) / h
    return diagonal, lower, upper


def _axis_factors(diagonal, lower, upper):
    # D^-1 A D is symmetric, with off-diagonal -sqrt(lower * upper).
    log_d = np.r_[0.0, np.cumsum(0.5 * (np.log(-lower) - np.log(-upper)))]
    d = np.exp(log_d - (log_d.max() + log_d.min()) / 2.0)
    eigenvalues, q = eigh_tridiagonal(diagonal, -np.sqrt(lower * upper))
    return eigenvalues, d[:, None] * q, q.T / d


def _sensor_design(study):
    axes = [_axis_factors(*_axis_stencil(study['L' + axis], study['kappa_' + axis],
                                       study['vel_' + axis], study['robin_' + axis]))
            for axis in ('x', 'y')]
    (ex, px, ix), (ey, py, iy) = axes
    modes = np.asarray(study['source_modes'], dtype=float)
    transformed = ix @ modes @ iy.T
    base = ex[:, None] + ey[None, :]
    si, sj = np.asarray(study['sensor_ij']).T
    samples = []
    for mu in study['mu_shifts']:
        fields = px @ (transformed / (base + mu)) @ py.T
        samples.append(fields[:, si, sj].T)
    return np.vstack(samples), fields


def _ridge_residualize(design, target, nuisance, ridge):
    values = np.column_stack((design, target))
    z = np.asarray(nuisance, dtype=float)
    if ridge > 0.0:
        coefficients = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ values)
    else:
        coefficients = np.linalg.lstsq(z, values, rcond=None)[0]
    residuals = values - z @ coefficients
    return residuals[:, :-1], residuals[:, -1]


def _box_simplex(design, target, ridge, cap, budget):
    n = design.shape[1]
    if budget == 0.0:
        return np.zeros(n)
    if budget == n * cap:
        return np.full(n, cap)

    # Minimize on the current face in least-squares form, not normal equations.
    # The anchor makes every search direction exactly budget-tangential.
    a = np.vstack((design, np.sqrt(ridge) * np.eye(n)))
    rhs = np.r_[target, np.zeros(n)]
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(50 * n):
        free = np.flatnonzero(status == 0)
        high = np.flatnonzero(status == 1)
        if free.size:
            face_budget = budget - cap * high.size
            anchor = free[-1]
            other = free[:-1]
            candidate = w.copy()
            if other.size:
                columns = a[:, other] - a[:, [anchor]]
                residual = rhs - cap * a[:, high].sum(axis=1) - face_budget * a[:, anchor]
                candidate[other] = np.linalg.lstsq(columns, residual, rcond=None)[0]
            candidate[anchor] = face_budget - candidate[other].sum()
            lower = free[candidate[free] < -1e-13]
            upper = free[candidate[free] > cap + 1e-13]
            if lower.size or upper.size:
                hits = np.r_[lower, upper]
                bounds = np.r_[np.zeros(lower.size), np.full(upper.size, cap)]
                steps = (bounds - w[hits]) / (candidate[hits] - w[hits])
                i = np.argmin(steps)
                w = np.clip(w + max(0.0, steps[i]) * (candidate - w), 0.0, cap)
                w[hits[i]] = bounds[i]
                status[hits[i]] = -1 if i < lower.size else 1
                continue
            w = np.clip(candidate, 0.0, cap)

        g = design.T @ (design @ w - target) + ridge * w
        # The best possible first-order transfer goes from the largest
        # gradient with removable mass to the smallest gradient with room.
        receivers = np.flatnonzero(status != 1)
        donors = np.flatnonzero(status != -1)
        receive = receivers[np.argmin(g[receivers])]
        donate = donors[np.argmax(g[donors])]
        if g[donate] - g[receive] <= 1e-12:
            return w
        status[[receive, donate]] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, target, ridge, cap, budget, w):
    residual = design @ w - target
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    g = design.T @ residual + ridge * w
    # A budget-indexed gradient order statistic maximizes the tangent bound.
    if cap == 0.0:
        nu = float(np.min(g))
    else:
        pivot = min(len(w) - 1, int(budget / cap))
        nu = float(np.partition(g, pivot)[pivot])
    lower = np.maximum(g - nu, 0.0)
    upper = np.maximum(nu - g, 0.0)
    # Stable equivalent of J - g@w + nu*budget + cap*sum(min(g-nu,0)).
    gap = float(lower @ w + upper @ (cap - w) + nu * (w.sum() - budget))
    return {'weights': w, 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
            'primal_objective': float(objective),
            'dual_lower_bound': float(objective - max(0.0, gap))}


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    """Fit each budget and reconstruct its middle-budget terminal field."""
    samples, terminal = _sensor_design(study)
    design, target = _ridge_residualize(samples, study['y_obs'], study['Z_nuisance'],
                                        float(study['nuisance_ridge']))
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    budgets = [float(b) for b in study['budget_grid']]
    weights = np.stack([_box_simplex(design, target, ridge, cap, b) for b in budgets])
    certificates = [_certificate(design, target, ridge, cap, b, w)
                    for b, w in zip(budgets, weights)]
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal),
            'kkt_certificates': certificates}
