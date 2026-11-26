"""Cell-centered Robin diffusion and ridge-profiled source tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(kappa, sigma, lengths, gamma):
    shape = kappa.shape
    spacing = np.asarray(lengths, dtype=float) / shape
    volume = np.prod(spacing)
    cells = np.arange(kappa.size).reshape(shape)
    diagonal = volume * np.broadcast_to(sigma, shape).astype(float)
    rows, cols, values = [], [], []
    for axis, h in enumerate(spacing):
        area = volume / h
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        kl, kr = kappa[left], kappa[right]
        face = (area / h) * (2 * kl * kr / (kl + kr))
        diagonal[left] += face
        diagonal[right] += face
        i, j, v = cells[left].ravel(), cells[right].ravel(), face.ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-v, -v))
        for side in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = side
            boundary = tuple(boundary)
            kb = kappa[boundary]
            diagonal[boundary] += area * gamma * kb / (kb + 0.5 * h * gamma)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal.ravel())
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(kappa.size, kappa.size),
    ).tocsc()
    return matrix, volume


def _profile(sensors, nuisance, observations, nuisance_ridge, source_ridge):
    # Augment by the nuisance penalty, then project onto the orthogonal
    # complement of the augmented nuisance columns. This also handles rank
    # deficiency when the nuisance penalty is zero.
    p = nuisance.shape[1]
    m = sensors.shape[1]
    augmented_z = np.vstack((nuisance, np.sqrt(nuisance_ridge) * np.eye(p)))
    data = np.hstack((sensors, observations))
    data = np.vstack((data, np.zeros((p, data.shape[1]))))
    if p:
        u, singular, _ = np.linalg.svd(augmented_z, full_matrices=False)
        threshold = np.finfo(float).eps * max(augmented_z.shape) * singular[0]
        span = u[:, singular > threshold]
        data -= span @ (span.T @ data)
    X, Y = data[:, :m], data[:, m:]
    norm_sq = np.einsum('ij,ij->j', X, X)
    single = np.divide(X.T @ Y, norm_sq[:, None],
                       out=np.zeros((m, Y.shape[1])), where=norm_sq[:, None] > 0)
    joint = np.linalg.lstsq(
        np.vstack((X, np.sqrt(source_ridge) * np.eye(m))),
        np.vstack((Y, np.zeros((m, Y.shape[1])))), rcond=None,
    )[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve every source basis, then fit signed single and joint weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    matrix, volume = _assemble(kappa, problem['sigma'],
                               [problem[key] for key in ('Lx', 'Ly', 'Lz')],
                               problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = volume * sources.reshape(sources.shape[0], -1).T
    # The stencil is SPD. Symmetric minimum-degree ordering limits fill,
    # and the numeric factorization is shared by all right-hand sides.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                  options={'SymmetricMode': True})
    basis = factor.solve(rhs).T.reshape(sources.shape)
    sensors = basis[(slice(None), *np.asarray(problem['sensor_ijk']).T)].T
    single, joint = _profile(sensors, np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {
        'basis_fields': basis,
        'single_source_profiled_amplitudes': single,
        'joint_source_weights': joint,
    }
