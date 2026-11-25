"""Cell-centered tensor Darcy forward model and robust source inversion."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _assemble(problem):
    """Return the integrated cell balance matrix and ambient load."""
    kxx, kxy, kyy, sigma = (np.asarray(problem[key], dtype=float)
                            for key in ('k_xx', 'k_xy', 'k_yy', 'sigma'))
    nx, ny = kxx.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    n = nx * ny
    cells = np.arange(n).reshape(nx, ny)

    def differences(lo, hi):
        lo, hi = lo.ravel(), hi.ravel()
        face = np.arange(lo.size)
        return sparse.csr_matrix((np.tile([1., -1.], (lo.size, 1)).ravel(),
                                  (np.repeat(face, 2), np.column_stack((lo, hi)).ravel())),
                                 shape=(lo.size, n))

    ix, iy = np.arange(nx), np.arange(ny)
    gx = differences(cells[np.maximum(ix - 1, 0), :],
                     cells[np.minimum(ix + 1, nx - 1), :])
    gy = differences(cells[:, np.maximum(iy - 1, 0)],
                     cells[:, np.minimum(iy + 1, ny - 1)])
    xlo, xhi = cells[:-1, :], cells[1:, :]
    ylo, yhi = cells[:, :-1], cells[:, 1:]
    dx, dy = differences(xlo, xhi), differences(ylo, yhi)
    kx0, kx1 = kxx[:-1, :], kxx[1:, :]
    ky0, ky1 = kyy[:, :-1], kyy[:, 1:]
    normal_x = (hy / hx) * (2 * kx0 * kx1 / (kx0 + kx1))
    normal_y = (hx / hy) * (2 * ky0 * ky1 / (ky0 + ky1))
    cross_x = (kxy[:-1, :] + kxy[1:, :]) / 8
    cross_y = (kxy[:, :-1] + kxy[:, 1:]) / 8
    fx = dx.multiply(normal_x.reshape(-1, 1)) + (
        gy[xlo.ravel()] + gy[xhi.ravel()]).multiply(cross_x.reshape(-1, 1))
    fy = dy.multiply(normal_y.reshape(-1, 1)) + (
        gx[ylo.ravel()] + gx[yhi.ravel()]).multiply(cross_y.reshape(-1, 1))

    gamma = problem['gamma_robin']
    robin = np.zeros_like(kxx)
    for i in (0, -1):
        robin[i, :] += hy * gamma * kxx[i, :] / (gamma * hx / 2 + kxx[i, :])
    for j in (0, -1):
        robin[:, j] += hx * gamma * kyy[:, j] / (gamma * hy / 2 + kyy[:, j])
    area = hx * hy
    matrix = dx.T @ fx + dy.T @ fy + sparse.diags((robin + area * sigma).ravel())
    return matrix.tocsc(), (robin * problem['u_ambient']).ravel(), area


def _forward(problem):
    matrix, ambient, area = _assemble(problem)
    basis = np.asarray(problem['basis'], dtype=float)
    loads = np.column_stack((ambient, area * basis.reshape(len(basis), matrix.shape[0]).T))
    return splu(matrix).solve(loads)


def _robust_fit(design, target, alpha):
    """Estimate sources and a free intercept, rejecting gross sensor spikes."""
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, np.sqrt(alpha)), 0.])

    def fit(weights):
        root = np.sqrt(weights)
        return np.linalg.lstsq(np.vstack((root[:, None] * design, penalty)),
                               np.r_[root * target, np.zeros(len(penalty))], rcond=None)[0]

    # Start from a robust location/scale before fitting the weak source responses.
    centered = target - np.median(target)
    scale = max(1.4826 * np.median(np.abs(centered)), 1e-12)
    weights = 1 / (1 + (centered / (4 * scale))**2)
    for _ in range(20):
        coef = fit(weights)
        residual = target - design @ coef
        center = np.median(residual)
        scale = max(1.4826 * np.median(np.abs(residual - center)), 1e-12)
        # With a clear gap, use the full quadratic information in the inliers.
        magnitude = np.abs(residual - center)
        ordered = np.sort(magnitude)
        first = max(design.shape[1] + 1, (3 * len(target) + 3) // 4)
        for split in range(first, len(target)):
            if ordered[split] > max(0.1, 12 * ordered[split - 1], 12 * scale):
                clean = magnitude <= ordered[split - 1]
                return fit(clean)
        # Give ordinary noise quadratic loss; taper only large residuals.
        updated = 1 / (1 + ((residual - center) / (5 * scale))**4)
        if np.max(np.abs(updated - weights)) < 1e-6:
            break
        weights = updated
    return fit(updated)


def infer_tensor_darcy_sources(problem: dict) -> dict:
    """Infer source weights, a common sensor offset, and the pressure field."""
    shape = np.shape(problem['k_xx'])
    responses = _forward(problem)
    sensors = np.ravel_multi_index(np.asarray(problem['sensor_ij']).T, shape)
    observed = responses[sensors]
    design = np.column_stack((observed[:, 1:], np.ones(len(sensors))))
    target = np.asarray(problem['y_obs'], dtype=float) - observed[:, 0]
    coef = _robust_fit(design, target, problem['tikhonov_alpha'])
    field = responses[:, 0] + responses[:, 1:] @ coef[:-1]
    return {'source_weights': coef[:-1], 'sensor_offset': float(coef[-1]),
            'predicted_field': field.reshape(shape)}
