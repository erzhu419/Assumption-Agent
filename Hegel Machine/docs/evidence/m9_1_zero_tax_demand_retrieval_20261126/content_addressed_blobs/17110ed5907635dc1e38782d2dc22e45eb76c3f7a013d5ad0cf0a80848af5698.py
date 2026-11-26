"""Cell-centered Robin diffusion and ridge-profiled source tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_matrix(kappa, sigma, lengths, gamma):
    """Assemble integrated finite-volume balances in C order."""
    shape = kappa.shape
    steps = np.asarray(lengths, dtype=float) / shape
    volume = float(np.prod(steps))
    cells = np.arange(kappa.size).reshape(shape)
    diagonal = (volume * np.broadcast_to(sigma, shape)).copy().ravel()
    rows, cols, values = [], [], []
    for axis, step in enumerate(steps):
        area = volume / step
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        i, j = cells[left].ravel(), cells[right].ravel()
        kl, kr = kappa[left].ravel(), kappa[right].ravel()
        conductance = (area / step) * (2 * kl * kr / (kl + kr))
        diagonal[i] += conductance
        diagonal[j] += conductance
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-conductance, -conductance))
        for end in (0, -1):
            face = [slice(None)] * 3
            face[axis] = end
            face = tuple(face)
            kb = kappa[face].ravel()
            diagonal[cells[face].ravel()] += area * gamma * kb / (kb + gamma * step / 2)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal)
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(kappa.size, kappa.size)).tocsc()
    return matrix, volume


def _profile_sources(S, Z, Y, nuisance_ridge, source_ridge):
    """Solve ridge objectives as augmented least squares, without normal equations."""
    n, m = S.shape
    p = Z.shape[1]
    z_aug = np.vstack((Z, np.sqrt(nuisance_ridge) * np.eye(p)))
    s_aug = np.vstack((S, np.zeros((p, m))))
    y_aug = np.vstack((Y, np.zeros((p, Y.shape[1]))))
    single = np.empty((m, Y.shape[1]))
    for i in range(m):
        design = np.column_stack((z_aug, s_aug[:, i]))
        single[i] = np.linalg.lstsq(design, y_aug, rcond=None)[0][-1]
    design = np.block([[z_aug, s_aug],
                       [np.zeros((m, p)), np.sqrt(source_ridge) * np.eye(m)]])
    target = np.vstack((y_aug, np.zeros((m, Y.shape[1]))))
    joint = np.linalg.lstsq(design, target, rcond=None)[0][p:]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Return unit-source fields and nuisance-profiled source estimates."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    matrix, volume = _robin_matrix(kappa, problem['sigma'],
                                   [problem[key] for key in ('Lx', 'Ly', 'Lz')],
                                   problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = np.asfortranarray(volume * sources.reshape(sources.shape[0], -1).T)
    # Positive definite stencil: preserve symmetric ordering and reuse one LU.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    fields = factor.solve(rhs).T.reshape(sources.shape)
    sensors = tuple(np.asarray(problem['sensor_ijk']).T)
    S = fields[(slice(None),) + sensors].T
    single, joint = _profile_sources(S, np.asarray(problem['nuisance_design']),
                                    np.asarray(problem['Y_obs']),
                                    problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
