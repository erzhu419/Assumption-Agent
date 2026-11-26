"""Finite-volume Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.linalg import orth, qr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, cg, spilu, splu


def _assemble(kappa, sigma, gamma, hx, hy):
    """Assemble the integrated cell balances in C order."""
    cells = np.arange(kappa.size).reshape(kappa.shape)
    diag = (hx * hy) * sigma

    # Each interior face adds a symmetric two-cell flux block.
    wx = (hy / hx) * (2 * kappa[:-1, :] * kappa[1:, :] /
                       (kappa[:-1, :] + kappa[1:, :]))
    wy = (hx / hy) * (2 * kappa[:, :-1] * kappa[:, 1:] /
                       (kappa[:, :-1] + kappa[:, 1:]))
    diag[:-1, :] += wx
    diag[1:, :] += wx
    diag[:, :-1] += wy
    diag[:, 1:] += wy

    # Half-cell diffusion and wall dissipation act in series.
    for i in (0, -1):
        wall = kappa[i, :]
        diag[i, :] += hy * gamma * wall / (wall + 0.5 * hx * gamma)
    for j in (0, -1):
        wall = kappa[:, j]
        diag[:, j] += hx * gamma * wall / (wall + 0.5 * hy * gamma)

    lo = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    hi = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    flux = -np.concatenate((wx.ravel(), wy.ravel()))
    return coo_matrix(
        (np.concatenate((diag.ravel(), flux, flux)),
         (np.concatenate((cells.ravel(), lo, hi)),
          np.concatenate((cells.ravel(), hi, lo)))),
        shape=(cells.size, cells.size),
    ).tocsr()


def _solve_fields(matrix, rhs):
    """Reuse one SPD incomplete-factor preconditioner for all columns."""
    order = reverse_cuthill_mckee(matrix, symmetric_mode=True)
    a = matrix[order, :][:, order].tocsr()
    ilu = spilu(a.tocsc(), permc_spec='NATURAL', diag_pivot_thresh=0,
                drop_tol=1e-5, fill_factor=8,
                options={'SymmetricMode': True, 'Equil': False})
    # Dropping need not preserve symmetry. Use L D L^T, not the raw ILU
    # solve, to keep the preconditioner symmetric positive definite for CG.
    # Factor the triangular matrix once to use native sparse solves in
    # both directions, without a Python-level row loop on each CG step.
    lower = splu(ilu.L, permc_spec='NATURAL', diag_pivot_thresh=0,
                 options={'Equil': False})
    inv_diag = 1.0 / np.abs(ilu.U.diagonal())

    def precondition(r):
        return lower.solve(inv_diag * lower.solve(r), trans='T')

    preconditioner = LinearOperator(a.shape, matvec=precondition, dtype=a.dtype)
    solutions = np.empty_like(rhs)
    for m, b in enumerate(rhs):
        x, info = cg(a, b[order], M=preconditioner, atol=0.0, rtol=1e-11)
        if info != 0:
            raise RuntimeError(f'Conjugate gradient did not converge for source {m}: {info}')
        solutions[m, order] = x
    return solutions


def _profile(s, z, y, lam_z, lam_s):
    """Eliminate the nuisance coefficients in the augmented observation space."""
    n_nuisance = z.shape[1]
    n_sources = s.shape[1]
    # Augmentation turns the nuisance ridge term into an ordinary squared norm.
    nuisance = np.vstack((z, np.sqrt(lam_z) * np.eye(n_nuisance)))
    s_aug = np.vstack((s, np.zeros((n_nuisance, n_sources))))
    y_aug = np.vstack((y, np.zeros((n_nuisance, y.shape[1]))))
    if lam_z > 0:
        q, _ = qr(nuisance, mode='economic')
    else:
        q = orth(nuisance)
    s_profiled = s_aug - q @ (q.T @ s_aug)
    y_profiled = y_aug - q @ (q.T @ y_aug)

    numerator = s_profiled.T @ y_profiled
    norm_sq = np.einsum('ij,ij->j', s_profiled, s_profiled)
    amplitudes = np.divide(numerator, norm_sq[:, None],
                           out=np.zeros_like(numerator), where=norm_sq[:, None] > 0)
    joint_design = np.vstack((s_profiled, np.sqrt(lam_s) * np.eye(n_sources)))
    joint_obs = np.vstack((y_profiled, np.zeros((n_sources, y.shape[1]))))
    weights = np.linalg.lstsq(joint_design, joint_obs, rcond=None)[0]
    return amplitudes, weights


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the source responses and fit single-source and joint models."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    matrix = _assemble(kappa, sigma, problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny)
    fields = _solve_fields(matrix, rhs).reshape(sources.shape)

    i, j = np.asarray(problem['sensor_ij']).T
    amplitudes, weights = _profile(fields[:, i, j].T,
                                   np.asarray(problem['nuisance_design'], dtype=float),
                                   np.asarray(problem['Y_obs'], dtype=float),
                                   problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}
