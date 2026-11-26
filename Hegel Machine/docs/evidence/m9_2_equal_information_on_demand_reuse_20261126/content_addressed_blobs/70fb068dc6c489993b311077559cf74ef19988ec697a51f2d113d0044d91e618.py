"""Complex upwind advection inversion with exhaustive coupled ridge selection."""

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _basis_responses(problem):
    basis = np.asarray(problem['basis'], dtype=np.float64)
    m, nx, ny = basis.shape
    hx = problem['Lx'] / (nx + 1)
    hy = problem['Ly'] / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    vx = np.asarray(problem['vx'], dtype=np.float64) / hx
    vy = np.asarray(problem['vy'], dtype=np.float64) / hy
    diagonal = (2 * dx + 2 * dy + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma'], dtype=np.float64)
                + 1j * problem['omega'])
    node = np.arange(nx * ny).reshape(nx, ny)
    rows = (node, node[1:, :], node[:-1, :], node[:, 1:], node[:, :-1])
    cols = (node, node[:-1, :], node[1:, :], node[:, :-1], node[:, 1:])
    values = (diagonal,
              -dx - np.maximum(vx[1:, :], 0),
              -dx + np.minimum(vx[:-1, :], 0),
              -dy - np.maximum(vy[:, 1:], 0),
              -dy + np.minimum(vy[:, :-1], 0))
    operator = coo_matrix(
        (np.concatenate([a.ravel() for a in values]),
         (np.concatenate([a.ravel() for a in rows]),
          np.concatenate([a.ravel() for a in cols]))),
        shape=(nx * ny, nx * ny), dtype=np.complex128).tocsc()
    rhs = basis.reshape(m, nx * ny).T.astype(np.complex128)
    return splu(operator).solve(rhs).reshape(nx, ny, m)


_CUTOFF = 1e-8


def _disk_fit(design, target):
    """Minimize the profiled quadratic over a product of complex unit disks."""
    adjoint = design.conj().T
    linear = adjoint @ target
    norms = np.sum(np.abs(design)**2, axis=0)
    scale = max(np.max(norms), np.max(np.abs(linear)))
    n = len(linear)
    if scale == 0:
        return np.zeros(n, dtype=np.complex128)

    def project(z):
        return z / np.maximum(1., np.abs(z))

    def value_grad(z):
        x = z[:n] + 1j*z[n:]
        dx = design @ x
        v = (0.5*np.vdot(dx, dx).real - np.vdot(linear, x).real)/scale
        g = (adjoint @ dx - linear)/scale
        return v, np.r_[g.real, g.imag]

    def disk(z):
        return 1. - z[:n]**2 - z[n:]**2

    def disk_jac(z):
        return -2.*np.hstack((np.diag(z[:n]), np.diag(z[n:])))

    # A shifted least-squares objective avoids losing small disk corrections
    # in the constant residual energy. The tangent gap is zero at a minimizer.
    x = np.zeros(n, dtype=np.complex128)
    for iteration in range(1024):
        residual = target - design @ x
        for j in range(n):
            if norms[j] == 0:
                continue
            update = x[j] + np.vdot(design[:, j], residual)/norms[j]
            update /= max(1., abs(update))
            residual -= design[:, j]*(update - x[j])
            x[j] = update
        gradient = (adjoint @ (design @ x) - linear)/scale
        gap = np.sum(np.abs(gradient)) + np.vdot(gradient, x).real
        if gap <= 2e-13:
            break
        if iteration == 3:
            opt = minimize(value_grad, np.r_[x.real, x.imag], jac=True,
                           method='SLSQP',
                           constraints={'type': 'ineq', 'fun': disk, 'jac': disk_jac},
                           options={'ftol': 2e-14, 'maxiter': 160})
            candidate = project(opt.x[:n] + 1j*opt.x[n:])
            if value_grad(np.r_[candidate.real, candidate.imag])[0] < value_grad(np.r_[x.real, x.imag])[0]:
                x = candidate
    # Round just inside the boundary so conversion to float64 weights cannot
    # turn an inactive disk constraint into an activation penalty.
    return project(x) * (_CUTOFF * (1. - 4.*np.finfo(float).eps))


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Recover complex weights and return the objective and field magnitude."""
    responses = _basis_responses(problem)
    i, j = np.asarray(problem['sensor_ij'], dtype=np.intp).T
    g = responses[i, j, :]
    y = (np.asarray(problem['y_obs_re'], dtype=np.float64)
         + 1j * np.asarray(problem['y_obs_im'], dtype=np.float64))
    lam0, lam2 = problem['lam0'], problem['lam2']
    m = g.shape[1]
    edges = [(int(u), int(v), float(gamma))
             for u, v, gamma in problem['pair_penalties']]

    def penalty(mask):
        return lam0*mask.bit_count() + sum(
            gamma for u,v,gamma in edges if mask & (1 << u) and mask & (1 << v))

    # QR compresses augmented ridge least squares to m rows; the discarded
    # residual component is constant over all masks.
    q, r = qr(np.vstack((g, np.sqrt(lam2)*np.eye(m))), mode='economic', check_finite=False)
    b = q[:len(y)].conj().T @ y
    best_w = np.zeros(m, dtype=np.complex128)
    best = 0.5*np.vdot(b,b).real
    masks = sorted((penalty(mask), mask) for mask in range(1 << m))
    for cost, mask in masks:
        if cost > best:
            break
        free = [k for k in range(m) if mask & (1 << k)]
        bounded = [k for k in range(m) if not mask & (1 << k)]
        if free:
            uq, ur = qr(r[:, free], mode='full', check_finite=False)
            rot = uq.conj().T @ np.column_stack((b, r[:, bounded]))
        else:
            rot = np.column_stack((b, r))
        k = len(free)
        target, design = rot[k:, 0], rot[k:, 1:]
        base = 0.5*np.vdot(target,target).real
        gain = _CUTOFF*np.sum(np.abs(design.conj().T @ target))
        # The tangent at zero lower-bounds the objective on all the disks.
        if cost + base - gain > best + 32*np.finfo(float).eps*(base + gain + best):
            continue
        w = np.zeros(m, dtype=np.complex128)
        if bounded:
            w[bounded] = _disk_fit(_CUTOFF*design, target)
        if free:
            w[free] = solve_triangular(ur[:k], rot[:k, 0] - rot[:k, 1:] @ w[bounded], check_finite=False)
        # Free coordinates need not exceed the threshold. If they do not,
        # their actual cost only decreases. Every weight vector is feasible
        # for the relaxed mask matching its actual active coordinates.
        actual = sum(1 << j for j, v in enumerate(np.abs(w)) if v > _CUTOFF)
        residual = r @ w - b
        score = 0.5*np.vdot(residual,residual).real + penalty(actual)
        if score < best:
            best, best_w = score, w

    residual = g @ best_w - y
    actual = sum(1 << j for j, v in enumerate(np.abs(best_w)) if v > _CUTOFF)
    objective = (0.5*np.vdot(residual,residual).real
                 + 0.5*lam2*np.vdot(best_w,best_w).real + penalty(actual))
    return {
        'weights_real': best_w.real.copy(),
        'weights_imag': best_w.imag.copy(),
        'objective': float(objective),
        'predicted_field_abs': np.abs(responses @ best_w),
    }
