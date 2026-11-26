"""Shifted transport inversion and box-simplex KKT certificates."""

import numpy as np


def _axis_spectrum(length, kappa, velocity, robin):
    """Factor the upwind/Robin operator as D Q diag(eigenvalues) Q.T D^-1."""
    kappa = np.asarray(kappa, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    h = length / kappa.size
    diffusion = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    left = np.maximum(velocity, 0) / h
    right = np.maximum(-velocity, 0) / h
    lower = diffusion + left[1:]
    upper = diffusion + right[:-1]
    diagonal = np.r_[0., diffusion] + np.r_[diffusion, 0.] + left + right
    diagonal[[0, -1]] += (robin / h) / (1 + robin * h / (2 * kappa[[0, -1]]))
    off = -np.sqrt(lower * upper)
    values, basis = np.linalg.eigh(np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1))
    logs = np.r_[0., np.cumsum(0.5 * (np.log(lower) - np.log(upper)))]
    scale = np.exp(logs - (logs.min() + logs.max()) / 2)
    return values, basis * scale[:, None], basis.T / scale[None, :]


def _responses(study):
    ex, vx, ix = _axis_spectrum(study['Lx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    ey, vy, iy = _axis_spectrum(study['Ly'], study['kappa_y'], study['vel_y'], study['robin_y'])
    modes = ix @ np.asarray(study['source_modes'], dtype=float) @ iy.T
    eigen_sum = ex[:, None] + ey[None, :]
    si, sj = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = vx @ (modes / (eigen_sum + shift)) @ vy.T
        blocks.append(fields[:, si, sj].T)
    return np.vstack(blocks), fields


def _profile(study, response):
    z = np.asarray(study['Z_nuisance'], dtype=float)
    joint = np.column_stack((response, np.asarray(study['y_obs']).ravel()))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    residual = joint - z @ np.linalg.solve(gram, z.T @ joint)
    return residual[:, :-1], residual[:, -1]


def _box_simplex(hessian, linear, cap, budget):
    """Minimize the quadratic by solving equality-constrained faces exactly."""
    n = linear.size
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((hessian @ w + linear).max())

    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1 lower, +1 upper
    for _ in range(100 * n):
        free = np.flatnonzero(active == 0)
        if not free.size:
            g = hessian @ w + linear
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] - g[i] <= 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue

        fixed = np.flatnonzero(active)
        rhs = -(linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed])
        face = hessian[np.ix_(free, free)]
        a, b = np.linalg.solve(face, np.column_stack((rhs, np.ones(free.size)))).T
        nu = (budget - w[fixed].sum() - a.sum()) / b.sum()
        trial = a + nu * b
        current = w[free]
        below, above = trial < -1e-13 * cap, trial > cap + 1e-13 * cap
        if below.any() or above.any():
            step = np.full(free.size, np.inf)
            step[below] = current[below] / (current[below] - trial[below])
            step[above] = (cap - current[above]) / (trial[above] - current[above])
            k = step.argmin()
            w[free] = np.clip(current + step[k] * (trial - current), 0., cap)
            active[free[k]] = -1 if below[k] else 1
            w[free[k]] = 0. if below[k] else cap
            continue

        w[free] = np.clip(trial, 0., cap)
        violation = active * (hessian @ w + linear - nu)
        k = violation.argmax()
        if violation[k] <= 1e-12:
            return w, float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active-set solve did not converge')


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    """Solve the transport inverse problem at every supplied budget."""
    response, terminal = _responses(study)
    design, target = _profile(study, response)
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    linear = -(design.T @ target)
    certs = []
    for budget in map(float, study['budget_grid']):
        w, nu = _box_simplex(hessian, linear, cap, budget)
        residual = design @ w - target
        objective = 0.5 * float(residual @ residual + ridge * (w @ w))
        gradient = design.T @ residual + ridge * w
        lower = np.maximum(gradient - nu, 0.)
        upper = np.maximum(nu - gradient, 0.)
        # Evaluate the supporting-plane gap as nonnegative box slack terms
        # plus the budget residual, avoiding subtraction of large objectives.
        gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - budget)
        dual = min(objective, float(objective - gap))
        certs.append({'weights': w, 'nu_sum': nu, 'lam_lower': lower,
                      'lam_upper': upper, 'primal_objective': objective,
                      'dual_lower_bound': dual})
    weights = np.stack([cert['weights'] for cert in certs])
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal),
            'kkt_certificates': certs}
