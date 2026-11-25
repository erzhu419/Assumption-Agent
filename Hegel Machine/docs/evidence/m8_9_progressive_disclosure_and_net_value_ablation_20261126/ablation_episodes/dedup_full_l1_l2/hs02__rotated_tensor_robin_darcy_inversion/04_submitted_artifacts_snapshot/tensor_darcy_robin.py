"""Finite-volume tensor Darcy source inversion with robust sensor fitting."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(problem):
    """Return the integrated cell operator and ambient boundary load."""
    kxx = np.asarray(problem['k_xx'], dtype=float)
    kyy = np.asarray(problem['k_yy'], dtype=float)
    kxy = np.asarray(problem['k_xy'], dtype=float)
    nx, ny = kxx.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    ids = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []

    def add(row, col, val):
        rows.append(row.ravel())
        cols.append(col.ravel())
        vals.append(val.ravel())

    def face(left, right, conductance, cross, up0, down0, up1, down1):
        # Every face contributes equal and opposite integrated outflows.
        for col, coefficient in ((left, conductance), (right, -conductance),
                                 (up0, -cross), (down0, cross),
                                 (up1, -cross), (down1, cross)):
            add(left, col, coefficient)
            add(right, col, -coefficient)

    yp = np.minimum(np.arange(ny) + 1, ny - 1)
    ym = np.maximum(np.arange(ny) - 1, 0)
    xp = np.minimum(np.arange(nx) + 1, nx - 1)
    xm = np.maximum(np.arange(nx) - 1, 0)
    face(ids[:-1], ids[1:], (2 * hy / hx) * kxx[:-1] * kxx[1:] / (kxx[:-1] + kxx[1:]),
         (kxy[:-1] + kxy[1:]) / 8, ids[:-1, yp], ids[:-1, ym], ids[1:, yp], ids[1:, ym])
    face(ids[:, :-1], ids[:, 1:], (2 * hx / hy) * kyy[:, :-1] * kyy[:, 1:] / (kyy[:, :-1] + kyy[:, 1:]),
         (kxy[:, :-1] + kxy[:, 1:]) / 8, ids[xp, :-1], ids[xm, :-1], ids[xp, 1:], ids[xm, 1:])

    gamma = problem['gamma_robin']
    wall = np.zeros_like(kxx)
    for i in (0, -1):
        wall[i] += hy * gamma * kxx[i] / (0.5 * hx * gamma + kxx[i])
    for j in (0, -1):
        wall[:, j] += hx * gamma * kyy[:, j] / (0.5 * hy * gamma + kyy[:, j])
    area = hx * hy
    add(ids, ids, wall + area * np.asarray(problem['sigma']))
    operator = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ids.size, ids.size)).tocsc()
    return operator, (wall * problem['u_ambient']).ravel(), area


def _responses(problem):
    operator, ambient, area = _assemble(problem)
    basis = np.asarray(problem['basis'], dtype=float)
    m, nx, ny = basis.shape
    rhs = np.column_stack((ambient, area * basis.reshape(m, nx * ny).T))
    return splu(operator).solve(rhs).reshape(nx, ny, m + 1)


def _fit_sources(design, target, alpha):
    """Robustly screen sensor spikes, then fit the retained observations."""
    n, p = design.shape
    ridge = np.sqrt(alpha) * np.eye(p)[:-1]
    zeros = np.zeros(p - 1)

    def fit(weight):
        root = np.sqrt(weight)
        return np.linalg.lstsq(np.vstack((root[:, None] * design, ridge)),
                               np.r_[root * target, zeros], rcond=None)[0]

    coef = np.zeros(p)
    coef[-1] = np.median(target)
    if p < n <= 3 * p:
        # With few redundant rows, a single high-leverage spike can pull
        # the initial fit far enough to hide itself from residual weights.
        # Compare leave-one-out starts on the full set by median residual.
        best = np.median(np.abs(target - design @ coef))
        keep = np.ones(n, dtype=bool)
        for k in range(n):
            keep[k] = False
            candidate = fit(keep)
            keep[k] = True
            residual = target - design @ candidate
            shift = np.median(residual)
            score = np.median(np.abs(residual - shift))
            if score < best:
                best = score
                candidate[-1] += shift
                coef = candidate
    floor = 1e-12 * max(1., np.max(np.abs(target)))
    for _ in range(40):
        residual = target - design @ coef
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        update = fit(1 / (1 + (residual / (2.5 * scale)) ** 2))
        if np.linalg.norm(update - coef) <= 1e-9 * (1 + np.linalg.norm(coef)):
            coef = update
            break
        coef = update
    residual = target - design @ coef
    median = np.median(residual)
    scale = max(floor, 1.4826 * np.median(np.abs(residual - median)))
    keep = np.abs(residual - median) <= 8 * scale
    return fit(keep) if np.count_nonzero(keep) >= p else coef


def infer_tensor_darcy_sources(problem: dict) -> dict:
    """Recover source amplitudes, a common sensor offset, and the pressure grid."""
    response = _responses(problem)
    i, j = np.asarray(problem['sensor_ij']).T
    sensors = response[i, j]
    design = np.column_stack((sensors[:, 1:], np.ones(len(sensors))))
    target = np.asarray(problem['y_obs'], dtype=float) - sensors[:, 0]
    coef = _fit_sources(design, target, float(problem['tikhonov_alpha']))
    return {'source_weights': coef[:-1], 'sensor_offset': float(coef[-1]),
            'predicted_field': response[..., 0] + response[..., 1:] @ coef[:-1]}
