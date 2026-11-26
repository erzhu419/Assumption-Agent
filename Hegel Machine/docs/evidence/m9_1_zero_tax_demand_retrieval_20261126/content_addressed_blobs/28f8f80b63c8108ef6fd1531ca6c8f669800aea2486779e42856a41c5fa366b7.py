"""Finite-volume Robin diffusion solves and ridge-profiled source fits."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _finite_volume_matrix(kappa, sigma, hx, hy, gamma):
    """Assemble one symmetric flux contribution per interior face."""
    nx, ny = kappa.shape
    ids = np.arange(kappa.size).reshape(nx, ny)
    diag = np.asarray(sigma, dtype=float).copy() * (hx * hy)

    k0, k1 = kappa[:-1, :], kappa[1:, :]
    wx = (hy / hx) * (2 * k0 * k1 / (k0 + k1))
    diag[:-1, :] += wx
    diag[1:, :] += wx
    k0, k1 = kappa[:, :-1], kappa[:, 1:]
    wy = (hx / hy) * (2 * k0 * k1 / (k0 + k1))
    diag[:, :-1] += wy
    diag[:, 1:] += wy

    diag[0, :] += hy * gamma * kappa[0, :] / (kappa[0, :] + 0.5 * hx * gamma)
    diag[-1, :] += hy * gamma * kappa[-1, :] / (kappa[-1, :] + 0.5 * hx * gamma)
    diag[:, 0] += hx * gamma * kappa[:, 0] / (kappa[:, 0] + 0.5 * hy * gamma)
    diag[:, -1] += hx * gamma * kappa[:, -1] / (kappa[:, -1] + 0.5 * hy * gamma)

    a = np.r_[ids[:-1, :].ravel(), ids[:, :-1].ravel()]
    b = np.r_[ids[1:, :].ravel(), ids[:, 1:].ravel()]
    d = ids.ravel()
    off = -np.r_[wx.ravel(), wy.ravel()]
    return coo_matrix((np.r_[diag.ravel(), off, off],
                       (np.r_[d, a, b], np.r_[d, b, a])),
                      shape=(d.size, d.size)).tocsc()


def _profile(S, Z, Y, lam_z, lam_s):
    # In the augmented space, nuisance ridge is ordinary least squares.
    # Eliminate its column space before either fit; this avoids normal equations.
    p = Z.shape[1]
    if p:
        Z_aug = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        U, singular, _ = np.linalg.svd(Z_aug, full_matrices=False)
        tol = np.finfo(float).eps * max(Z_aug.shape) * singular[0]
        U = U[:, singular > tol]
        S = np.vstack((S, np.zeros((p, S.shape[1]))))
        Y = np.vstack((Y, np.zeros((p, Y.shape[1]))))
        S -= U @ (U.T @ S)
        Y -= U @ (U.T @ Y)

    denom = np.einsum('km,km->m', S, S)[:, None]
    numer = S.T @ Y
    amplitudes = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    if lam_s:
        S = np.vstack((S, np.sqrt(lam_s) * np.eye(S.shape[1])))
        Y = np.vstack((Y, np.zeros((S.shape[1], Y.shape[1]))))
    weights = np.linalg.lstsq(S, Y, rcond=None)[0]
    return amplitudes, weights


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return all candidate PDE fields and both nuisance-profiled fits."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    A = _finite_volume_matrix(kappa, problem['sigma'], hx, hy, problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny).T
    # The grid is small enough for a sparse direct factorization, shared by all modes.
    fields = splu(A, permc_spec='MMD_AT_PLUS_A').solve(rhs).T.reshape(sources.shape)
    sensors = np.asarray(problem['sensor_ij'])
    S = fields[:, sensors[:, 0], sensors[:, 1]].T
    amplitudes, weights = _profile(S, np.asarray(problem['nuisance_design'], dtype=float),
                                   np.asarray(problem['Y_obs'], dtype=float),
                                   problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}
