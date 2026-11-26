"""Complex upwind PDE inversion with exhaustive pairwise-penalized selection."""

import numpy as np
from scipy.optimize import minimize, root
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _solve_fields(problem, basis):
    m, nx, ny = basis.shape
    n = nx * ny
    hx = problem.get('Lx', 1.0) / (nx + 1)
    hy = problem.get('Ly', 1.0) / (ny + 1)
    dx, dy = problem['nu'] / hx**2, problem['nu'] / hy**2
    vx = np.asarray(problem['vx'], dtype=np.float64).ravel() / hx
    vy = np.asarray(problem['vy'], dtype=np.float64).ravel() / hy
    ids = np.arange(n)
    i, j = ids // ny, ids % ny
    diagonal = (2 * (dx + dy) + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma']).ravel() + 1j * problem['omega'])
    rows, cols, values = [ids], [ids], [diagonal]
    for keep, step, coefficient in (
        (i > 0, -ny, -dx - np.maximum(vx, 0)),
        (i < nx - 1, ny, -dx + np.minimum(vx, 0)),
        (j > 0, -1, -dy - np.maximum(vy, 0)),
        (j < ny - 1, 1, -dy + np.minimum(vy, 0)),
    ):
        rows.append(ids[keep])
        cols.append(ids[keep] + step)
        values.append(coefficient[keep])
    operator = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n), dtype=np.complex128,
    ).tocsc()
    return splu(operator).solve(basis.reshape(m, n).T)




def _polish_disk_fit(design, target, u):
    """Refine the active disk phases after profiling out interior coordinates."""
    boundary = np.abs(u) >= 1 - 1e-7
    if not np.any(boundary):
        return u
    free = design[:, ~boundary]
    q, r = np.linalg.qr(free, mode='reduced')
    edges = design[:, boundary]
    cross = q.conj().T @ edges
    center = q.conj().T @ target
    edges = edges - q @ cross
    target_perp = target - q @ center
    gram = edges.conj().T @ edges
    rhs = edges.conj().T @ target_perp

    if len(rhs) == 1:
        if rhs[0] == 0:
            return u
        z = rhs / np.abs(rhs)
    else:
        def phase_equations(theta):
            z = np.exp(1j * theta)
            grad = z.conj() * (gram @ z - rhs)
            jac = (z.conj()[:, None] * gram * z).real
            jac[np.diag_indices_from(jac)] -= grad.real
            return grad.imag, jac

        result = root(phase_equations, np.angle(u[boundary]), jac=True, tol=1e-12)
        z = np.exp(1j * result.x)

    candidate = u.copy()
    candidate[boundary] = z
    candidate[~boundary] = np.linalg.solve(r, center - cross @ z)
    if np.any(np.abs(candidate) > 1 + 8 * np.finfo(float).eps):
        return u
    candidate /= np.maximum(1, np.abs(candidate))
    step = design @ (candidate - u)
    improvement = np.vdot(design @ u - target + 0.5 * step, step).real
    return candidate if improvement < 0 else u

def _fit_inactive(design, target):
    """Minimize the projected residual on the product of unit complex disks."""
    k = design.shape[1]
    # An interior optimum is an ordinary ridge least-squares fit. Solve
    # directly instead of relying on an objective-gap test at weak signal.
    radius = (1 - 4 * np.finfo(float).eps) * 1e-8
    u = np.linalg.lstsq(design, target, rcond=0.0)[0]
    if np.all(np.abs(u) <= 1):
        return radius * u
    gram = design.conj().T @ design
    rhs = design.conj().T @ target
    energy = gram.diagonal().real
    u /= np.maximum(1, np.abs(u))
    # Only the target component seen by the disk design affects progress.
    tol = 2e-14 * np.sum(np.abs(rhs))
    for _ in range(64):
        for j in range(k):
            if energy[j] > 0:
                dot = rhs[j] - gram[j] @ u + energy[j] * u[j]
                u[j] = dot / max(energy[j], abs(dot))
        gradient = gram @ u - rhs
        gap = np.vdot(gradient, u).real + np.abs(gradient).sum()
        if gap <= tol:
            break
    else:
        # Refine correlated disk coordinates together when sweeps stall.
        scale = max(np.linalg.norm(gram), np.linalg.norm(rhs))

        def objective(x):
            z = x[:k] + 1j * x[k:]
            h = gram @ z
            value = np.vdot(z, 0.5 * h - rhs).real / scale
            grad = (h - rhs) / scale
            return value, np.r_[grad.real, grad.imag]

        eye = np.eye(k)
        result = minimize(objective, np.r_[u.real, u.imag], jac=True,
                          method='SLSQP', constraints={
                              'type': 'ineq',
                              'fun': lambda x: 1 - x[:k]**2 - x[k:]**2,
                              'jac': lambda x: -2 * np.hstack((eye * x[:k], eye * x[k:])),
                          }, options={'ftol': 1e-13, 'maxiter': 150})
        candidate = result.x[:k] + 1j * result.x[k:]
        candidate /= np.maximum(1, np.abs(candidate))
        if objective(np.r_[candidate.real, candidate.imag])[0] < objective(np.r_[u.real, u.imag])[0]:
            u = candidate
    u = _polish_disk_fit(design, target, u)
    # Keep the projected boundary on the uncharged side in floating point.
    return radius * u / np.maximum(1, np.abs(u))

def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return the best complex ridge support, its objective and field magnitude."""
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    m, nx, ny = basis.shape
    fields = _solve_fields(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=np.intp)
    design = fields[sensors[:, 0] * ny + sensors[:, 1], :]
    observed = (np.asarray(problem['y_obs_re'], dtype=np.float64)
                + 1j * np.asarray(problem['y_obs_im'], dtype=np.float64))
    lam0, lam2 = problem['lam0'], problem['lam2']
    pairs = [(1 << int(u) | 1 << int(v), gamma)
             for u, v, gamma in problem['pair_penalties']]

    # Compress the augmented ridge system once. Its orthogonal residual
    # is independent of the support and need not enter the comparison.
    augmented = np.vstack((design, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:len(observed)].conj().T @ observed
    threshold = 1e-8

    def penalty(mask):
        return lam0 * mask.bit_count() + sum(
            gamma for bits, gamma in pairs if (mask & bits) == bits
        )

    def active_mask(w):
        return sum(1 << k for k in np.flatnonzero(np.abs(w) > threshold).tolist())

    best = 0.5 * np.vdot(target, target).real
    weights = np.zeros(m, dtype=np.complex128)
    for mask in range(1 << m):
        support = [k for k in range(m) if mask & (1 << k)]
        inactive = [k for k in range(m) if not mask & (1 << k)]
        charge = penalty(mask)
        if charge >= best:
            continue
        qa, ra = np.linalg.qr(r[:, support], mode='reduced')
        projected_target = qa.conj().T @ target
        perpendicular = target - qa @ projected_target
        fitted = np.zeros(m, dtype=np.complex128)
        fitted[support] = np.linalg.solve(ra, projected_target)
        base = 0.5 * np.vdot(perpendicular, perpendicular).real
        value = base + penalty(active_mask(fitted))
        if value < best:
            best, weights = value, fitted.copy()
        if not inactive:
            continue
        # Eliminate free coordinates first. Discarding their activation
        # lower limits is valid: any that become inactive only reduce charge.
        other = r[:, inactive]
        cross = qa.conj().T @ other
        projected = other - qa @ cross
        # The maximum linear improvement over the inactive disks is a
        # bound because the remaining quadratic term is nonnegative.
        gain = threshold * np.abs(projected.conj().T @ perpendicular).sum()
        if base - gain + charge >= best:
            continue
        fitted[inactive] = _fit_inactive(threshold * projected, perpendicular)
        fitted[support] = np.linalg.solve(ra, projected_target - cross @ fitted[inactive])
        err = r @ fitted - target
        value = 0.5 * np.vdot(err, err).real + penalty(active_mask(fitted))
        if value < best:
            best, weights = value, fitted

    residual = design @ weights - observed
    objective = (0.5 * np.vdot(residual, residual).real
                 + 0.5 * lam2 * np.vdot(weights, weights).real
                 + penalty(active_mask(weights)))
    return {
        'weights_real': weights.real.copy(),
        'weights_imag': weights.imag.copy(),
        'objective': float(objective),
        'predicted_field_abs': np.abs(fields @ weights).reshape(nx, ny),
    }
