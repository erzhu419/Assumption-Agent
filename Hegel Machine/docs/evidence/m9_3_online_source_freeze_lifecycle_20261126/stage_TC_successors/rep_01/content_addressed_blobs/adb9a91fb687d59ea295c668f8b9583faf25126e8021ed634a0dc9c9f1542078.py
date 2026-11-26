"""Shifted separable transport and tangent-certified budget inversion."""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _axis(length, diffusivity, velocity, robin):
    k = np.asarray(diffusivity, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face + np.minimum(v[:-1], 0) / h
    diagonal = np.zeros_like(k)
    diagonal[1:] -= lower
    diagonal[:-1] -= upper
    # The Robin resistance is in series with diffusion across a half-cell.
    diagonal[0] += robin / (h * (1 + robin * h / (2 * k[0]))) + max(v[0], 0) / h
    diagonal[-1] += robin / (h * (1 + robin * h / (2 * k[-1]))) - min(v[-1], 0) / h
    return sparse.diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _transport(study):
    a, b = (_axis(study['L' + d], study['kappa_' + d], study['vel_' + d],
                  study['robin_' + d]) for d in ('x', 'y'))
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    # C-order flattening sends A @ U + U @ B.T to A ⊗ I + I ⊗ B.
    op = (sparse.kron(a, sparse.eye(ny), format='csc') +
          sparse.kron(sparse.eye(nx), b, format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = np.asfortranarray(modes.reshape(m, nx * ny).T)
    i, j = np.asarray(study['sensor_ij']).T
    readings = []
    for shift in study['mu_shifts']:
        field = splu(op + float(shift) * identity).solve(rhs)
        readings.append(field[i * ny + j, :])
    return np.concatenate(readings, axis=0), field.T.reshape(m, nx, ny)


def _profile(design, study):
    z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((design, np.asarray(study['y_obs']).ravel()))
    # Residualize both blocks, rather than taking a square root of the
    # ridge residual operator: the contract is a norm of these residuals.
    if z.shape[1]:
        ridge = float(study['nuisance_ridge'])
        if ridge == 0:
            coef = np.linalg.lstsq(z, values, rcond=None)[0]
        else:
            coef = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ values)
        values -= z @ coef
    return values[:, :-1], values[:, -1]


def _dual_parts(g, cap, budget):
    # A budget-weighted gradient quantile maximizes the tangent lower bound.
    index = min(max(int(budget / cap), 0), g.size - 1) if cap else 0
    nu = float(np.partition(g, index)[index])
    return nu, np.maximum(g - nu, 0), np.maximum(nu - g, 0)


def _budget_qp(s, y, ridge, cap, budget):
    n = s.shape[1]
    h = s.T @ s + ridge * np.eye(n)
    q = -(s.T @ y)
    w = np.full(n, budget / n)
    if 0 < budget < n * cap:
        def objective(x):
            return float(x @ (0.5 * (h @ x) + q))

        fit = minimize(objective, w, jac=lambda x: h @ x + q,
                       bounds=[(0.0, cap)] * n,
                       constraints={'type': 'eq', 'fun': lambda x: x.sum() - budget,
                                    'jac': lambda x: np.ones(n)},
                       method='SLSQP', options={'ftol': 1e-14, 'maxiter': 200})
        w = np.clip(fit.x, 0, cap)
        # Restore the sum to roundoff before evaluating the tangent gap.
        delta = budget - w.sum()
        i = np.argmax(cap - w if delta > 0 else w)
        w[i] += delta

        # Polish by exact two-coordinate minimization on a feasible line.
        # The tangent gap, not SLSQP's status, controls termination.
        for _ in range(10000):
            g = h @ w + q
            _, lower, upper = _dual_parts(g, cap, budget)
            if lower @ w + upper @ (cap - w) <= 1e-11:
                break
            i = np.argmin(np.where(w < cap, g, np.inf))
            j = np.argmax(np.where(w > 0, g, -np.inf))
            if g[j] <= g[i]:
                break
            curvature = h[i, i] + h[j, j] - 2 * h[i, j]
            room = min(cap - w[i], w[j])
            step = min(room, (g[j] - g[i]) / curvature) if curvature > 0 else room
            w[i] = min(cap, w[i] + step)
            w[j] = max(0.0, w[j] - step)
    elif budget != 0:
        w.fill(cap)

    residual = s @ w - y
    primal = float(0.5 * (residual @ residual) + 0.5 * ridge * (w @ w))
    g = s.T @ residual + ridge * w
    nu, lower, upper = _dual_parts(g, cap, budget)
    bound = primal - float(g @ w) + nu * budget - cap * float(upper.sum())
    return {'weights': w, 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
            'primal_objective': primal, 'dual_lower_bound': min(primal, bound)}


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    """Solve the shifted PDEs, then certify a mixture at each supplied budget."""
    design, terminal = _transport(study)
    s, y = _profile(design, study)
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    certs = [_budget_qp(s, y, ridge, cap, float(b)) for b in study['budget_grid']]
    weights = np.stack([c['weights'] for c in certs])
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.tensordot(weights[1], terminal, axes=1),
            'kkt_certificates': certs}
