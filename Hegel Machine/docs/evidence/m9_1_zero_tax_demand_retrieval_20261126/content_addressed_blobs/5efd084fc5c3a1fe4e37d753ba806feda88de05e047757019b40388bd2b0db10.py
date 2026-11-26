"""Cell-centered Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble_operator(kappa, sigma, gamma, hx, hy):
    """Assemble shared face conductances and cell-local dissipation."""
    nx, ny = kappa.shape
    cell = np.arange(nx * ny).reshape(nx, ny)
    kx0, kx1 = kappa[:-1, :], kappa[1:, :]
    ky0, ky1 = kappa[:, :-1], kappa[:, 1:]
    gx = (hy / hx) * (2 * kx0 * kx1 / (kx0 + kx1))
    gy = (hx / hy) * (2 * ky0 * ky1 / (ky0 + ky1))

    diag = sigma * (hx * hy)
    diag[:-1, :] += gx
    diag[1:, :] += gx
    diag[:, :-1] += gy
    diag[:, 1:] += gy
    for row in (0, nx - 1):
        kb = kappa[row, :]
        diag[row, :] += hy * gamma * kb / (kb + gamma * (0.5 * hx))
    for col in (0, ny - 1):
        kb = kappa[:, col]
        diag[:, col] += hx * gamma * kb / (kb + gamma * (0.5 * hy))

    left = np.concatenate((cell[:-1, :].ravel(), cell[:, :-1].ravel()))
    right = np.concatenate((cell[1:, :].ravel(), cell[:, 1:].ravel()))
    off = -np.concatenate((gx.ravel(), gy.ravel()))
    main = cell.ravel()
    return coo_matrix(
        (np.concatenate((diag.ravel(), off, off)),
         (np.concatenate((main, left, right)), np.concatenate((main, right, left)))),
        shape=(cell.size, cell.size),
    ).tocsc()


def _profile_sources(samples, nuisance, observations, nuisance_ridge, source_ridge):
    """Project out nuisance in the augmented least-squares space."""
    p = nuisance.shape[1]
    augmented = np.concatenate((samples, observations), axis=1)
    if p:
        z = np.vstack((nuisance, np.sqrt(nuisance_ridge) * np.eye(p)))
        augmented = np.pad(augmented, ((0, p), (0, 0)))
        if nuisance_ridge > 0:
            q, _ = np.linalg.qr(z, mode='reduced')
        else:
            # Without the ridge, only project onto the actual nuisance span.
            q, singular, _ = np.linalg.svd(z, full_matrices=False)
            q = q[:, singular > np.finfo(float).eps * max(z.shape) * singular[0]]
        augmented -= q @ (q.T @ augmented)

    m = samples.shape[1]
    s, y = augmented[:, :m], augmented[:, m:]
    energy = np.einsum('ij,ij->j', s, s)
    cross = s.T @ y
    single = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                       where=energy[:, None] > 0)
    joint = np.linalg.lstsq(
        np.vstack((s, np.sqrt(source_ridge) * np.eye(m))),
        np.pad(y, ((0, m), (0, 0))), rcond=None,
    )[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all candidate fields and fit single and joint source coefficients."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    operator = _assemble_operator(kappa, sigma, problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny).T
    # Factor once and solve against all candidate right-hand sides together.
    fields = splu(operator, permc_spec='MMD_AT_PLUS_A').solve(rhs).T.reshape(sources.shape)

    sensor_ij = np.asarray(problem['sensor_ij'])
    samples = fields[:, sensor_ij[:, 0], sensor_ij[:, 1]].T
    single, joint = _profile_sources(
        samples, np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float), problem['nuisance_ridge'], problem['source_ridge'],
    )
    return dict(basis_fields=fields, single_source_profiled_amplitudes=single,
                joint_source_weights=joint)
