"""Complex Dirichlet advection inversion with exact support enumeration."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    basis = np.asarray(problem['basis'], dtype=np.float64)
    m, nx, ny = basis.shape
    hx = problem['Lx'] / (nx + 1)
    hy = problem['Ly'] / (ny + 1)
    nu = problem['nu']
    dx, dy = nu / hx**2, nu / hy**2
    vx = np.asarray(problem['vx'], dtype=np.float64) / hx
    vy = np.asarray(problem['vy'], dtype=np.float64) / hy
    ids = np.arange(nx * ny).reshape(nx, ny)
    diagonal = (2 * (dx + dy) + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma']) + 1j * problem['omega'])

    # Every entry uses the velocity at its row, not at the neighboring node.
    bands = (
        (ids, ids, diagonal),
        (ids[1:, :], ids[:-1, :], -dx - np.maximum(vx[1:, :], 0)),
        (ids[:-1, :], ids[1:, :], -dx + np.minimum(vx[:-1, :], 0)),
        (ids[:, 1:], ids[:, :-1], -dy - np.maximum(vy[:, 1:], 0)),
        (ids[:, :-1], ids[:, 1:], -dy + np.minimum(vy[:, :-1], 0)),
    )
    rows, cols, values = (np.concatenate([a.ravel() for a in group])
                          for group in zip(*bands))
    operator = coo_matrix((values, (rows, cols)), shape=(nx * ny, nx * ny),
                          dtype=np.complex128).tocsc()
    fields = splu(operator).solve(basis.reshape(m, -1).T.astype(np.complex128))
    sensors = np.asarray(problem['sensor_ij'], dtype=np.intp)
    g = fields[ids[sensors[:, 0], sensors[:, 1]], :]
    y = np.asarray(problem['y_obs_re'], dtype=np.float64) + 1j * np.asarray(
        problem['y_obs_im'], dtype=np.float64)
    lam0, lam2 = problem['lam0'], problem['lam2']
    # Keep the ridge in separate rows: forming G^H G can round it away.
    design = np.vstack((g, np.sqrt(lam2) * np.eye(m)))
    target = np.concatenate((y, np.zeros(m)))
    pairs = [(int(u), int(v), gamma) for u, v, gamma in problem['pair_penalties']]

    best_obj = 0.5 * np.vdot(y, y).real
    best = np.zeros(m, dtype=np.complex128)
    cutoff = 1e-8
    # Leave one rounding margin when mapping a unit-disk solution back to w.
    radius = cutoff * (1 - 4 * np.finfo(float).eps)
    indices = np.arange(m)
    for mask in range(1 << m):
        allowed = (mask & (1 << indices)) != 0
        penalty = lam0 * np.count_nonzero(allowed)
        penalty += sum(gamma for u, v, gamma in pairs if allowed[u] and allowed[v])
        if penalty >= best_obj:
            continue
        active, inactive = indices[allowed], indices[~allowed]
        fixed = design[:, active]
        bounded = design[:, inactive]
        if active.size:
            q, r = np.linalg.qr(fixed, mode='reduced')
            projected_target = q.conj().T @ target
            reduced_target = target - q @ projected_target
        else:
            reduced_target = target

        # The tangent at zero gives a lower bound over all inactive disks.
        # No lower bound on active magnitudes is needed for this relaxation.
        c = bounded.conj().T @ reduced_target
        lower = 0.5 * np.vdot(reduced_target, reduced_target).real
        lower -= cutoff * np.abs(c).sum()
        if lower + penalty >= best_obj:
            continue

        w = np.zeros(m, dtype=np.complex128)
        if inactive.size:
            if active.size:
                projected_bounded = q.conj().T @ bounded
                bounded = bounded - q @ projected_bounded
            w[inactive] = radius * _unit_disk_fit(bounded, c, radius)
        if active.size:
            t = projected_target
            if inactive.size:
                t = t - projected_bounded @ w[inactive]
            w[active] = solve_triangular(r, t, check_finite=False)

        # If a relaxed active coefficient ended up inactive, its actual
        # penalty can only decrease. That smaller mask is also enumerated.
        charged = np.abs(w) > cutoff
        penalty = lam0 * np.count_nonzero(charged)
        penalty += sum(gamma for u, v, gamma in pairs if charged[u] and charged[v])
        residual = g @ w - y
        obj = 0.5 * (np.vdot(residual, residual).real + lam2 * np.vdot(w, w).real) + penalty
        if obj < best_obj:
            best_obj, best = obj, w

    return {
        'weights_real': best.real.copy(),
        'weights_imag': best.imag.copy(),
        'objective': float(best_obj),
        'predicted_field_abs': np.abs(fields @ best).reshape(nx, ny),
    }


def _unit_disk_fit(design, c, radius):
    """Minimize the profiled quadratic in complex unit-disk coordinates."""
    n = c.size
    h = radius * (design.conj().T @ design)
    scale = max(np.max(np.abs(c)), np.max(h.diagonal().real))
    if scale == 0:
        return np.zeros(n, dtype=np.complex128)
    h, c = h / scale, c / scale

    def objective(x):
        z = x[:n] + 1j * x[n:]
        hz = h @ z
        grad = hz - c
        return np.vdot(z, 0.5 * hz - c).real, np.r_[grad.real, grad.imag]

    def disk(x):
        return 1 - x[:n]**2 - x[n:]**2

    def disk_jac(x):
        return -2 * np.hstack((np.diag(x[:n]), np.diag(x[n:])))

    # Start at the separate one-coordinate optima; SLSQP handles coupling.
    denom = np.maximum(h.diagonal().real, np.abs(c))
    z = np.divide(c, denom, out=np.zeros_like(c), where=denom > 0)
    fit = minimize(objective, np.r_[z.real, z.imag], jac=True, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': disk, 'jac': disk_jac},
                   options={'ftol': 1e-14, 'maxiter': 200})
    z = fit.x[:n] + 1j * fit.x[n:]
    return z / np.maximum(1, np.abs(z))
