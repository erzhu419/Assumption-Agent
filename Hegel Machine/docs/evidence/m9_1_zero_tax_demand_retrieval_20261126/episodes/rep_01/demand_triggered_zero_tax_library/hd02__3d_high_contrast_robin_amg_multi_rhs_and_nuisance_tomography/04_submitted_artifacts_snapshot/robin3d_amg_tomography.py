"""Cell-centered Robin finite volumes and ridge-profiled source inversion."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _robin_operator(kappa, sigma, h, gamma):
    """Assemble the integrated cell balance in C-order."""
    vol = np.prod(h)
    ids = np.arange(kappa.size).reshape(kappa.shape)
    diagonal = (vol * sigma).copy()
    rows, cols, values = [], [], []
    for axis, step in enumerate(h):
        area = vol / step
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        kl, kr = kappa[left], kappa[right]
        g = (area / step) * (2.0 * kl * kr / (kl + kr))
        diagonal[left] += g
        diagonal[right] += g
        i, j, off = ids[left].ravel(), ids[right].ravel(), -g.ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((off, off))
        for endpoint in (0, -1):
            face = [slice(None)] * 3
            face[axis] = endpoint
            face = tuple(face)
            kb = kappa[face]
            diagonal[face] += area * gamma * kb / (kb + 0.5 * step * gamma)
    rows.append(ids.ravel())
    cols.append(ids.ravel())
    values.append(diagonal.ravel())
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(kappa.size, kappa.size)).tocsc()


def _profile_sources(S, Z, Y, lam_z, lam_s):
    """Project out the nuisance columns in the augmented ridge system."""
    m, p = S.shape[1], Z.shape[1]
    if p:
        Z_aug = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        S = np.pad(S, ((0, p), (0, 0)))
        Y = np.pad(Y, ((0, p), (0, 0)))
        U, singular, _ = np.linalg.svd(Z_aug, full_matrices=False)
        tol = np.finfo(float).eps * max(Z_aug.shape) * singular[0]
        Q = U[:, singular > tol]
        S = S - Q @ (Q.T @ S)
        Y = Y - Q @ (Q.T @ Y)
    denom = np.sum(S * S, axis=0)
    numer = S.T @ Y
    single = np.divide(numer, denom[:, None], out=np.zeros_like(numer),
                       where=denom[:, None] > 0)
    joint = np.linalg.lstsq(np.vstack((S, np.sqrt(lam_s) * np.eye(m))),
                            np.pad(Y, ((0, m), (0, 0))), rcond=None)[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve all candidate fields and fit independent and joint amplitudes."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    h = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    A = _robin_operator(kappa, sigma, h, problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m = len(sources)
    rhs = np.asfortranarray(sources.reshape(m, -1).T * np.prod(h))
    lu = splu(A)
    u = np.column_stack([lu.solve(b) for b in rhs.T])
    fields = u.T.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    single, joint = _profile_sources(
        fields[:, i, j, k].T,
        np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float),
        problem['nuisance_ridge'], problem['source_ridge'])
    return dict(basis_fields=fields, single_source_profiled_amplitudes=single,
                joint_source_weights=joint)
