"""Cell-centered 3D Robin diffusion and nuisance-profiled source inversion."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(kappa, sigma, spacing, gamma):
    """Assemble the integrated cell balance in C-order indexing."""
    volume = np.prod(spacing)
    cells = np.arange(kappa.size).reshape(kappa.shape)
    diagonal = (volume * sigma).copy()
    rows, cols, values = [], [], []
    for axis, h in enumerate(spacing):
        area = volume / h
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        kl, kr = kappa[lo], kappa[hi]
        conductance = (area / h) * (2 * kl * kr / (kl + kr))
        diagonal[lo] += conductance
        diagonal[hi] += conductance
        left, right = cells[lo].ravel(), cells[hi].ravel()
        rows.extend((left, right))
        cols.extend((right, left))
        values.extend((-conductance.ravel(), -conductance.ravel()))
        for edge in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = edge
            boundary = tuple(boundary)
            kb = kappa[boundary]
            diagonal[boundary] += area * gamma * kb / (kb + 0.5 * h * gamma)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal.ravel())
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(kappa.size, kappa.size)).tocsc()


def _profile(S, Z, Y, lam_z, lam_s):
    """Project out nuisance columns in the ridge-augmented observation space."""
    m, p = S.shape[1], Z.shape[1]
    data = np.column_stack((S, Y))
    if p:
        design = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        data = np.vstack((data, np.zeros((p, data.shape[1]))))
        nuisance = np.linalg.lstsq(design, data, rcond=None)[0]
        data -= design @ nuisance
    s, y = data[:, :m], data[:, m:]
    den = np.einsum('ij,ij->j', s, s)
    num = s.T @ y
    # A source fully in the nuisance span has no identifiable single amplitude.
    tol = (np.finfo(float).eps * max(data.shape) * np.linalg.norm(S, axis=0)) ** 2
    single = np.divide(num, den[:, None], out=np.zeros_like(num),
                       where=(den > tol)[:, None])
    if lam_s:
        s = np.vstack((s, np.sqrt(lam_s) * np.eye(m)))
        y = np.vstack((y, np.zeros((m, y.shape[1]))))
    joint = np.linalg.lstsq(s, y, rcond=None)[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve all candidate responses and fit the two profiled source models."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    spacing = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    A = _assemble(kappa, np.asarray(problem['sigma'], dtype=float), spacing,
                  problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = np.asfortranarray(np.prod(spacing) * sources.reshape(len(sources), -1).T)
    factor = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    solution = factor.solve(rhs)
    gap = rhs - A @ solution
    refine = np.linalg.norm(gap, axis=0) > 1e-8 * np.linalg.norm(rhs, axis=0)
    if np.any(refine):
        solution[:, refine] += factor.solve(gap[:, refine])
    fields = solution.T.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    single, joint = _profile(fields[:, i, j, k].T,
                             np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
