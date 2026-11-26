"""Separable reaction-transport inversion on a capped simplex."""

import numpy as np
from scipy.linalg import eigh_tridiagonal, helmert
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import splu


def _stencil(length, n, kappa, velocity, robin):
    h = length / n
    kappa = np.asarray(kappa, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    face = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    diag = np.abs(velocity) / h
    diag[:-1] += face
    diag[1:] += face
    for i in (0, n - 1):
        diag[i] += robin * kappa[i] / (h * (0.5 * robin * h + kappa[i]))
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    return np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)


def _axis_spectrum(matrix):
    # M = diag(d) @ Q @ diag(eigenvalues) @ Q.T @ diag(1/d).
    lower = -np.diag(matrix, -1)
    upper = -np.diag(matrix, 1)
    log_d = np.r_[0.0, np.cumsum(0.5 * (np.log(lower) - np.log(upper)))]
    if np.ptp(log_d) > 25:
        return None
    d = np.exp(log_d - (log_d.max() + log_d.min()) * 0.5)
    eigenvalues, q = eigh_tridiagonal(np.diag(matrix), -np.sqrt(lower) * np.sqrt(upper))
    return eigenvalues, d[:, None] * q, q.T / d


def _shifted_fields(ax, by, shifts, modes):
    xs, ys = _axis_spectrum(ax), _axis_spectrum(by)
    if xs is not None and ys is not None:
        ex, tx, rx = xs
        ey, ty, ry = ys
        modal_rhs = rx @ modes @ ry.T
        denom = ex[:, None] + ey[None, :] + shifts[:, None, None]
        fields = tx @ (modal_rhs[None, :, :, :] / denom[:, None, :, :]) @ ty.T
        residual = ax @ fields + fields @ by.T + shifts[:, None, None, None] * fields - modes
        scale = np.maximum(np.max(np.abs(modes), axis=(-2, -1)), np.finfo(float).tiny)
        if np.all(np.max(np.abs(residual), axis=(-2, -1)) <= 1e-12 * scale):
            return fields
    # A strongly nonnormal axis can make spectral transforms ill-conditioned.
    nx, ny = modes.shape[1:]
    op = kron(csr_matrix(ax), eye(ny)) + kron(eye(nx), csr_matrix(by))
    identity = eye(nx * ny, format='csc')
    rhs = modes.reshape(len(modes), -1).T.copy()
    return np.stack([splu((op + mu * identity).tocsc()).solve(rhs).T.reshape(modes.shape)
                     for mu in shifts])


def _multiplier(gradient, box, budget):
    # The slope changes from positive to nonpositive at this order statistic.
    i = min(len(gradient) - 1, max(0, int(np.ceil(budget / box)) - 1)) if box else 0
    return float(np.partition(gradient, i)[i])


def _solve_capped(design, target, box, budget):
    """Feasible active-set least squares, with equality eliminated on each face."""
    n = design.shape[1]
    w = np.full(n, budget / n)
    if budget <= 0 or budget >= n * box:
        return np.clip(w, 0, box)
    # -1: fixed at zero; +1: fixed at cap; 0: free.
    bound = np.zeros(n, dtype=np.int8)
    bound_tol = 8 * np.finfo(float).eps * max(box, budget)
    for _ in range(40 * n):
        free = np.flatnonzero(bound == 0)
        if free.size:
            trial = w.copy()
            budget_free = budget - box * np.count_nonzero(bound == 1)
            trial[free] += (budget_free - trial[free].sum()) / free.size
            # Solve for the shortest correction on the face. Recomputing a
            # minimum-norm weight vector can move along a flat direction and
            # immediately hit the same bound after releasing it.
            basis = helmert(free.size).T
            step, _, rank, _ = np.linalg.lstsq(design[:, free] @ basis,
                                                target - design @ trial, rcond=None)
            trial[free] += basis @ step
            # At an integer-budget vertex, subtraction can put a free
            # coordinate a few ulps outside its mathematically exact bound.
            near_bound = (bound == 0) & (trial >= -bound_tol) & (trial <= box + bound_tol)
            trial[near_bound] = np.clip(trial[near_bound], 0.0, box)
            delta = trial - w
            hits = np.flatnonzero((bound == 0) & ((trial < 0) | (trial > box)))
            if hits.size:
                limits = np.where(trial[hits] < 0, 0.0, box)
                fractions = (limits - w[hits]) / delta[hits]
                first = int(np.argmin(fractions))
                w = np.clip(w + np.clip(fractions[first], 0, 1) * delta, 0, box)
                i = hits[first]
                w[i] = limits[first]
                bound[i] = -1 if trial[i] < 0 else 1
                continue
            w = trial
        g = design.T @ (design @ w - target)
        tol = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(g)))
        if free.size:
            nu = np.mean(g[free])
            violation = bound * (g - nu)
            i = int(np.argmax(violation))
            if violation[i] <= tol:
                return w
            if rank < free.size - 1:
                # On a flat face, use the actual certificate rather than
                # treating roundoff in reduced costs as evidence of descent.
                nu = _multiplier(g, box, budget)
                gap = (np.maximum(g - nu, 0) @ w
                       + np.maximum(nu - g, 0) @ (box - w)
                       + nu * (w.sum() - budget))
                if abs(gap) <= 1e-9:
                    return w
            bound[i] = 0
        else:
            lo = np.flatnonzero(bound == -1)
            hi = np.flatnonzero(bound == 1)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] - g[i] <= tol:
                return w
            bound[[i, j]] = 0
    raise RuntimeError('Box-simplex active-set solver did not converge')


def solve_kronecker_box_simplex_source(study: dict) -> dict:
    """Recover source weights, their terminal field, and a tangent lower bound."""
    ax = _stencil(study['Lx'], study['nx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    by = _stencil(study['Ly'], study['ny'], study['kappa_y'], study['vel_y'], study['robin_y'])
    modes = np.asarray(study['source_modes'], dtype=float)
    shifts = np.asarray(study['mu_shifts'], dtype=float)
    fields = _shifted_fields(ax, by, shifts, modes)
    ix, iy = np.asarray(study['sensor_ij']).T
    stack = fields[:, :, ix, iy].transpose(0, 2, 1).reshape(-1, len(modes))
    z = np.asarray(study['Z_nuisance'], dtype=float)
    data = np.column_stack((np.asarray(study['y_obs']).reshape(-1), stack))
    residualized = data - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    target, s = residualized[:, 0], residualized[:, 1:]
    ridge = float(study['source_ridge'])
    box, budget = float(study['u_box']), float(study['total_budget'])
    design = np.vstack((s, np.sqrt(ridge) * np.eye(len(modes))))
    w = _solve_capped(design, np.r_[target, np.zeros(len(modes))], box, budget)
    residual = s @ w - target
    g = s.T @ residual + ridge * w
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    nu = _multiplier(g, box, budget)
    lower, upper = np.maximum(g - nu, 0), np.maximum(nu - g, 0)
    # Algebraically identical to the tangent formula, without cancellation at KKT.
    gap = float(lower @ w + upper @ (box - w) + nu * (w.sum() - budget))
    cert = {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': lower.tolist(),
            'lam_upper': upper.tolist(), 'primal_objective': primal,
            'dual_lower_bound': primal - max(0.0, gap)}
    return {'weights': w, 'terminal_field': np.einsum('m,mxy->xy', w, fields[-1]),
            'kkt_certificate': cert}
