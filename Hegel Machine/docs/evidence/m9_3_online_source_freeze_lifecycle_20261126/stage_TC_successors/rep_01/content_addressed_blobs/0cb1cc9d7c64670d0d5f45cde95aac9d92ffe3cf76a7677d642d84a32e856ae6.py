"""Shifted transport responses and certified box-simplex source fits."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis_operator(length, kappa, velocity, robin):
    kappa = np.asarray(kappa, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    h = length / kappa.size
    face = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    diag = np.r_[face, 0.] + np.r_[0., face] + np.abs(velocity) / h
    ends = [0, kappa.size - 1]
    diag[ends] += robin / (h * (1 + robin * h / (2 * kappa[ends])))
    lower = -face - np.maximum(velocity[1:], 0.) / h
    upper = -face - np.maximum(-velocity[:-1], 0.) / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _transport(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    ax, by = (_axis_operator(study['L' + a], study['kappa_' + a],
                            study['vel_' + a], study['robin_' + a]) for a in 'xy')
    # With C-order flattening, U -> Ax U + U By.T has this Kronecker sum.
    base = (sparse.kron(ax, sparse.eye(ny), format='csc') +
            sparse.kron(sparse.eye(nx), by, format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    i, j = np.asarray(study['sensor_ij']).T
    sensors = i * ny + j
    rhs = np.asfortranarray(sources.reshape(m, nx * ny).T)
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(base + float(shift) * eye).solve(rhs)
        blocks.append(fields[sensors, :])
    return np.concatenate(blocks, axis=0), fields


def _profile(design, target, nuisance, ridge):
    # The objective uses ordinary ridge residuals, not the square-root map.
    values = np.column_stack((design, target))
    if ridge == 0:
        coef = np.linalg.lstsq(nuisance, values, rcond=None)[0]
    else:
        gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
        coef = np.linalg.solve(gram, nuisance.T @ values)
    residual = values - nuisance @ coef
    return residual[:, :-1], residual[:, -1]


def _fit_budget(hessian, linear, cap, budget):
    """Feasible active-set QP: minimize on a face, then test its multipliers."""
    m = linear.size
    if budget == 0 or budget == m * cap:
        return np.full(m, budget / m)
    w = np.full(m, budget / m)
    status = np.zeros(m, dtype=np.int8)  # -1 lower, +1 upper, 0 free
    grad_tol = 32 * np.finfo(float).eps * max(
        np.max(np.abs(linear)), cap * np.max(np.sum(np.abs(hessian), axis=1)))
    for _ in range(100 * m):
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            g = hessian @ w + linear
            lo = np.where(status < 0, g, np.inf).argmin()
            hi = np.where(status > 0, g, -np.inf).argmax()
            if g[hi] <= g[lo] + grad_tol:
                return w
            status[[lo, hi]] = 0
            continue

        fixed = np.flatnonzero(status)
        # Stationarity on the free set, with the remaining budget as an equality.
        n = free.size
        system = np.ones((n + 1, n + 1))
        system[:n, :n] = hessian[np.ix_(free, free)]
        system[n, n] = 0.
        rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed],
                    budget - cap * np.count_nonzero(status > 0)]
        sol = np.linalg.solve(system, rhs)
        trial, nu = sol[:-1], -sol[-1]
        if np.any((trial < 0.) | (trial > cap)):
            old = w[free]
            delta = trial - old
            steps = np.full(n, np.inf)
            low = trial < 0.
            high = trial > cap
            steps[low] = -old[low] / delta[low]
            steps[high] = (cap - old[high]) / delta[high]
            k = steps.argmin()
            w[free] = np.clip(old + steps[k] * delta, 0., cap)
            w[free[k]] = 0. if low[k] else cap
            status[free[k]] = -1 if low[k] else 1
            continue

        w[free] = trial
        reduced = hessian @ w + linear - nu
        violation = status * reduced
        k = violation.argmax()
        if violation[k] <= grad_tol:
            return w
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, target, ridge, w, cap, budget):
    residual = design @ w - target
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = design.T @ residual + ridge * w
    # The minimizer of the tangent plane fills its cheapest coordinates first.
    # A gradient at the budget cutoff is a maximizing equality multiplier.
    k = min(int(budget / cap), w.size - 1) if cap > 0 else 0
    nu = float(np.partition(g, k)[k])
    lower = np.maximum(g - nu, 0.)
    upper = np.maximum(nu - g, 0.)
    gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - budget)
    return {'weights': w, 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
            'primal_objective': primal,
            'dual_lower_bound': float(primal - max(0., gap))}


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    sensor_design, terminal = _transport(study)
    design, target = _profile(sensor_design, np.asarray(study['y_obs']).ravel(),
                              np.asarray(study['Z_nuisance']), float(study['nuisance_ridge']))
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    m = design.shape[1]
    hessian = design.T @ design + ridge * np.eye(m)
    linear = -(design.T @ target)
    certs = []
    for budget in study['budget_grid']:
        budget = float(budget)
        w = _fit_budget(hessian, linear, cap, budget)
        certs.append(_certificate(design, target, ridge, w, cap, budget))
    weights = np.stack([cert['weights'] for cert in certs])
    field = (terminal @ weights[1]).reshape(np.shape(study['source_modes'])[1:])
    return {'weights_by_budget': weights, 'terminal_field_mid_budget': field,
            'kkt_certificates': certs}
