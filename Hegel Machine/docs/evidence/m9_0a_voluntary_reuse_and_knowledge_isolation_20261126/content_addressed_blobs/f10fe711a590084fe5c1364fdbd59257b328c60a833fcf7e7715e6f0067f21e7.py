"""Cell-centered Robin diffusion solve and ridge-profiled source matching."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_matrix(kappa, sigma, lx, ly, gamma):
    nx, ny = kappa.shape
    hx, hy = lx / nx, ly / ny
    cells = np.arange(nx * ny).reshape(nx, ny)
    wx = (hy / hx) * (2 * kappa[:-1, :] * kappa[1:, :] /
                      (kappa[:-1, :] + kappa[1:, :]))
    wy = (hx / hy) * (2 * kappa[:, :-1] * kappa[:, 1:] /
                      (kappa[:, :-1] + kappa[:, 1:]))

    diag = hx * hy * sigma
    diag[:-1, :] += wx
    diag[1:, :] += wx
    diag[:, :-1] += wy
    diag[:, 1:] += wy
    for i in (0, nx - 1):
        kb = kappa[i, :]
        diag[i, :] += hy * gamma * kb / (kb + 0.5 * hx * gamma)
    for j in (0, ny - 1):
        kb = kappa[:, j]
        diag[:, j] += hx * gamma * kb / (kb + 0.5 * hy * gamma)

    a = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    b = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    w = np.concatenate((wx.ravel(), wy.ravel()))
    d = cells.ravel()
    matrix = coo_matrix((np.concatenate((diag.ravel(), -w, -w)),
                         (np.concatenate((d, a, b)), np.concatenate((d, b, a)))),
                        shape=(d.size, d.size)).tocsc()
    return matrix, hx * hy


def _remove_nuisance(z, data, ridge):
    # Project in the augmented data space so the norm includes the nuisance penalty.
    p = z.shape[1]
    if not p:
        return data
    if ridge > 0:
        z_aug = np.vstack((z, np.sqrt(ridge) * np.eye(p)))
        data = np.vstack((data, np.zeros((p, data.shape[1]))))
        q, _ = np.linalg.qr(z_aug, mode='reduced')
    else:
        u, singular, _ = np.linalg.svd(z, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0] * max(z.shape) * np.finfo(float).eps)
        q = u[:, :rank]
    return data - q @ (q.T @ data)


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all source PDEs and return scalar profiles and joint source weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    matrix, area = _robin_matrix(kappa, sigma, problem['Lx'], problem['Ly'],
                                 problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m = sources.shape[0]
    rhs = area * sources.reshape(m, kappa.size).T
    # Sparse factorization is shared across all source right-hand sides.
    lu = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
              options={'SymmetricMode': True})
    basis = lu.solve(rhs).T.reshape(sources.shape)

    sensors = np.asarray(problem['sensor_ij'])
    s = basis[:, sensors[:, 0], sensors[:, 1]].T
    z = np.asarray(problem['nuisance_design'], dtype=float)
    y = np.asarray(problem['Y_obs'], dtype=float)
    profiled = _remove_nuisance(z, np.hstack((s, y)), problem['nuisance_ridge'])
    s, y = profiled[:, :m], profiled[:, m:]
    denom = np.sum(s * s, axis=0)[:, None]
    numer = s.T @ y
    single = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    ridge = problem['source_ridge']
    if ridge > 0:
        s = np.vstack((s, np.sqrt(ridge) * np.eye(m)))
        y = np.vstack((y, np.zeros((m, y.shape[1]))))
    joint = np.linalg.lstsq(s, y, rcond=None)[0]
    return {'basis_fields': basis,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
