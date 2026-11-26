"""Separable shifted transport and certified box-simplex budget optimization."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh_tridiagonal


def _axis_modes(length, kappa, velocity, robin):
    """Return eigenvalues and left/right transforms in the physical grid basis."""
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    faces = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    positive = np.maximum(v, 0.0) / h
    negative = np.maximum(-v, 0.0) / h
    diagonal = np.r_[0.0, faces] + np.r_[faces, 0.0] + positive + negative
    diagonal[0] += robin / h / (1 + robin * h / (2 * k[0]))
    diagonal[-1] += robin / h / (1 + robin * h / (2 * k[-1]))
    lower = faces + positive[1:]
    upper = faces + negative[:-1]
    # D^{-1} A D is symmetric when D[i+1]/D[i] = sqrt(lower[i]/upper[i]).
    log_d = np.r_[0.0, np.cumsum(0.5 * np.log(lower / upper))]
    d = np.exp(log_d - 0.5 * (log_d.max() + log_d.min()))
    eig, q = eigh_tridiagonal(diagonal, -np.sqrt(lower * upper))
    return eig, q.T / d, d[:, None] * q


def _responses(study):
    axes = [_axis_modes(study['L' + a], study['kappa_' + a],
                        study['vel_' + a], study['robin_' + a]) for a in ('x', 'y')]
    (ex, lx, rx), (ey, ly, ry) = axes
    sources = np.asarray(study['source_modes'], dtype=float)
    src_modes = lx @ sources @ ly.T
    eigen_sums = ex[:, None] + ey[None, :]
    si, sj = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = rx @ (src_modes / (eigen_sums + shift)) @ ry.T
        blocks.append(fields[:, si, sj].T)
    return np.concatenate(blocks, axis=0), fields


def _box_simplex(hessian, linear, cap, total):
    """Feasible active-set method: hit bounds, then release wrong-sign bounds."""
    n = linear.size
    if total == 0:
        return np.zeros(n), float(np.min(linear))
    if total == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(hessian @ w + linear))

    w = np.full(n, total / n)
    active = np.zeros(n, dtype=np.int8)  # -1: lower; +1: upper
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        grad = hessian @ w + linear
        if free.size == 0:
            lo = np.flatnonzero(active == -1)
            hi = np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j] <= grad[i] + 2e-13:
                return w, float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue

        fixed = np.flatnonzero(active != 0)
        rhs = linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed]
        factor = cho_factor(hessian[np.ix_(free, free)])
        x, s = cho_solve(factor, np.column_stack((-rhs, np.ones(free.size)))).T
        remaining = total - w[fixed].sum()
        nu = (remaining - x.sum()) / s.sum()
        candidate = x + nu * s
        candidate += (remaining - candidate.sum()) / free.size
        direction = candidate - w[free]
        crossing = np.full(free.size, np.inf)
        down = (candidate < 0) & (direction < 0)
        up = (candidate > cap) & (direction > 0)
        crossing[down] = -w[free[down]] / direction[down]
        crossing[up] = (cap - w[free[up]]) / direction[up]
        j = int(np.argmin(crossing))
        if crossing[j] < 1 and (candidate[j] < -2e-14 or candidate[j] > cap + 2e-14):
            w[free] = np.clip(w[free] + max(0.0, crossing[j]) * direction, 0, cap)
            active[free[j]] = 1 if up[j] else -1
            w[free[j]] = cap if up[j] else 0.0
            continue

        w[free] = np.clip(candidate, 0, cap)
        grad = hessian @ w + linear
        nu = float(np.mean(grad[free]))
        violation = active * (grad - nu)
        j = int(np.argmax(violation))
        if violation[j] <= 2e-13:
            return w, nu
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    stacked, terminal = _responses(study)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((stacked, study['y_obs']))
    ridge = float(study['nuisance_ridge'])
    # The contract uses the ridge residualizer itself, not its square root.
    gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
    profiled = values - nuisance @ cho_solve(cho_factor(gram), nuisance.T @ values)
    design, target = profiled[:, :-1], profiled[:, -1]
    lam, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = design.T @ design + lam * np.eye(design.shape[1])
    linear = -(design.T @ target)
    certificates = []
    for total in map(float, study['budget_grid']):
        w, nu = _box_simplex(hessian, linear, cap, total)
        residual = design @ w - target
        primal = 0.5 * float(residual @ residual + lam * (w @ w))
        g = design.T @ residual + lam * w
        lower = np.maximum(g - nu, 0.0)
        upper = np.maximum(nu - g, 0.0)
        # Same tangential lower bound, evaluated via the complementarity gap.
        gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - total)
        certificates.append({'weights': w, 'nu_sum': nu, 'lam_lower': lower,
                             'lam_upper': upper, 'primal_objective': primal,
                             'dual_lower_bound': primal - max(0.0, float(gap))})
    weights = np.stack([cert['weights'] for cert in certificates])
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal),
            'kkt_certificates': certificates}
