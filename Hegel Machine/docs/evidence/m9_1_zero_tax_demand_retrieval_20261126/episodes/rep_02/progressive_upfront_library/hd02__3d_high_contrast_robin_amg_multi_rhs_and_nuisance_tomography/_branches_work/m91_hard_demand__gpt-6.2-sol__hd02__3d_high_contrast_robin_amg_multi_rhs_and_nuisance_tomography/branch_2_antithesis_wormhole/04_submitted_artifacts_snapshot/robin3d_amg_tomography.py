"""Conservative 3D Robin diffusion-reaction and ridge-profiled tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_operator(kappa, sigma, h, gamma):
    """Assemble integrated cell balances in C order."""
    shape = kappa.shape
    n = kappa.size
    cells = np.arange(n).reshape(shape)
    volume = np.prod(h)
    diagonal = np.broadcast_to(volume * sigma, shape).copy()
    rows, cols, entries = [], [], []
    for axis, step in enumerate(h):
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        area = volume / step
        kl, kh = kappa[lo], kappa[hi]
        conductance = (area / step) * (2 * kl * kh / (kl + kh))
        diagonal[lo] += conductance
        diagonal[hi] += conductance
        left, right = cells[lo].ravel(), cells[hi].ravel()
        rows.extend((left, right))
        cols.extend((right, left))
        entries.extend((-conductance.ravel(), -conductance.ravel()))
        for edge in (0, -1):
            face = [slice(None)] * 3
            face[axis] = edge
            face = tuple(face)
            kb = kappa[face]
            diagonal[face] += area * gamma * kb / (kb + gamma * step / 2)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    entries.append(diagonal.ravel())
    matrix = coo_matrix((np.concatenate(entries),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n)).tocsc()
    return matrix, volume


def _basis_solve(matrix, rhs):
    # The operator is SPD: symmetric minimum-degree ordering limits 3D fill,
    # and diagonal pivots suffice. Reuse the factors for every source.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    fields = np.empty_like(rhs)
    for i, b in enumerate(rhs):
        u = factor.solve(b)
        threshold = 1e-7 * np.linalg.norm(b)
        for _ in range(2):
            residual = b - matrix @ u
            if np.linalg.norm(residual) <= threshold:
                break
            u += factor.solve(residual)
        else:
            if not np.linalg.norm(b - matrix @ u) <= threshold:
                raise ArithmeticError('Robin solve did not reach the residual tolerance')
        fields[i] = u
    return fields


def _profile(s, z, y, lam_z, lam_s):
    # Apply the orthogonal complement of [Z; sqrt(lam_z) I] to both
    # [S; 0] and [Y; 0]. The added rows retain the nuisance penalty.
    m, p = s.shape[1], z.shape[1]
    z_aug = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
    data = np.vstack((np.column_stack((s, y)), np.zeros((p, m + y.shape[1]))))
    if p:
        q, sv, _ = np.linalg.svd(z_aug, full_matrices=False)
        q = q[:, sv > np.finfo(float).eps * max(z_aug.shape) * sv[0]]
        data -= q @ (q.T @ data)
    sx, yx = data[:, :m], data[:, m:]
    energy = np.sum(sx * sx, axis=0)
    cross = sx.T @ yx
    # A column in the nuisance span has no identifiable single amplitude.
    noise_floor = (np.finfo(float).eps * max(z_aug.shape)) ** 2
    usable = energy > noise_floor * np.sum(s * s, axis=0)
    amplitudes = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                           where=usable[:, None])
    joint = np.linalg.lstsq(
        np.vstack((sx, np.sqrt(lam_s) * np.eye(m))),
        np.vstack((yx, np.zeros((m, y.shape[1])))), rcond=None)[0]
    return amplitudes, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Return source response fields, single-source amplitudes and joint weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    h = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    a, volume = _robin_operator(kappa, sigma, h, float(problem['gamma_robin']))
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = _basis_solve(a, volume * sources.reshape(sources.shape[0], -1))
    fields = fields.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    s = fields[:, i, j, k].T
    amplitudes, joint = _profile(s, np.asarray(problem['nuisance_design'], dtype=float),
                                 np.asarray(problem['Y_obs'], dtype=float),
                                 float(problem['nuisance_ridge']),
                                 float(problem['source_ridge']))
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': joint}
