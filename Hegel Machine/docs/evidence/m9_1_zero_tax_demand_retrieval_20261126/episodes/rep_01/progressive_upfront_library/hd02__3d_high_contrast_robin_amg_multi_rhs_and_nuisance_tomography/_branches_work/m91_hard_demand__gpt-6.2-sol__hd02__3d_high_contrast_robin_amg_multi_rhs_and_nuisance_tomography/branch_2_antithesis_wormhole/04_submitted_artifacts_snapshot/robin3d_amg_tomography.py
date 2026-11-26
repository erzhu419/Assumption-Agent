"""Conservative cell-centered Robin diffusion and ridge-profiled tomography."""

import numpy as np
from scipy import sparse
from scipy.linalg import lstsq
from scipy.sparse.linalg import splu


def _assemble(problem):
    kappa = np.asarray(problem['kappa'], dtype=float)
    h = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    volume = np.prod(h)
    gamma = problem['gamma_robin']
    index = np.arange(kappa.size).reshape(kappa.shape)
    diagonal = volume * np.array(problem['sigma'], dtype=float, copy=True)
    rows, cols, values = [], [], []
    for axis, step in enumerate(h):
        a = [slice(None)] * 3
        b = a.copy()
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        ka, kb = kappa[a], kappa[b]
        g = (volume / step**2) * (2.0 * ka * kb / (ka + kb))
        diagonal[a] += g
        diagonal[b] += g
        i, j = index[a].ravel(), index[b].ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-g.ravel(), -g.ravel()))
        face = [slice(None)] * 3
        for side in (0, -1):
            face[axis] = side
            f = tuple(face)
            k = kappa[f]
            diagonal[f] += (volume / step) * gamma * k / (k + 0.5 * step * gamma)
    flat = index.ravel()
    rows.append(flat)
    cols.append(flat)
    values.append(diagonal.ravel())
    matrix = sparse.csc_matrix((np.concatenate(values),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(kappa.size, kappa.size))
    return matrix, volume


def _solve_fields(matrix, rhs):
    # SPD permits symmetric ordering and diagonal pivots. Reuse the factors.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    fields = factor.solve(rhs)
    threshold = np.linalg.norm(rhs, axis=0) * 1e-7
    for _ in range(3):
        residual = rhs - matrix @ fields
        bad = ~(np.linalg.norm(residual, axis=0) <= threshold)
        if not np.any(bad):
            return fields
        fields[:, bad] += factor.solve(residual[:, bad])
    if not np.all(np.linalg.norm(rhs - matrix @ fields, axis=0) <= threshold):
        raise RuntimeError('Robin solve did not meet the PDE residual tolerance')
    return fields


def _profile(S, Z, Y, nuisance_ridge, source_ridge):
    # Ridge on gamma becomes extra rows. Orthogonal projection then eliminates
    # gamma without subtracting nearly equal Gram matrices.
    p = Z.shape[1]
    m = S.shape[1]
    V = np.column_stack((S, Y))
    if p:
        augmented_Z = np.vstack((Z, np.sqrt(nuisance_ridge) * np.eye(p)))
        V = np.pad(V, ((0, p), (0, 0)))
        V -= augmented_Z @ lstsq(augmented_Z, V)[0]
    X, targets = V[:, :m], V[:, m:]
    energy = np.sum(X**2, axis=0)[:, None]
    dot = X.T @ targets
    amplitudes = np.divide(dot, energy, out=np.zeros_like(dot), where=energy > 0)
    weights = lstsq(np.vstack((X, np.sqrt(source_ridge) * np.eye(m))),
                    np.pad(targets, ((0, m), (0, 0))))[0]
    return amplitudes, weights


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve all source fields, then fit independent and joint source models."""
    matrix, volume = _assemble(problem)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = volume * sources.reshape(sources.shape[0], -1).T
    fields = _solve_fields(matrix, rhs).T.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    amplitudes, weights = _profile(fields[:, i, j, k].T,
                                   np.asarray(problem['nuisance_design'], dtype=float),
                                   np.asarray(problem['Y_obs'], dtype=float),
                                   problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}
