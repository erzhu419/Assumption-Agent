"""Finite-volume Robin diffusion solver and ridge-profiled source fits."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, cg, spilu, splu, spsolve_triangular


def _robin_matrix(kappa, sigma, gamma, hx, hy):
    """Assemble the integrated cell balances in C-order."""
    ids = np.arange(kappa.size).reshape(kappa.shape)
    kx0, kx1 = kappa[:-1, :], kappa[1:, :]
    ky0, ky1 = kappa[:, :-1], kappa[:, 1:]
    tx = (hy / hx) * (2 * kx0 * kx1 / (kx0 + kx1))
    ty = (hx / hy) * (2 * ky0 * ky1 / (ky0 + ky1))

    diag = (hx * hy) * sigma
    diag[:-1, :] += tx
    diag[1:, :] += tx
    diag[:, :-1] += ty
    diag[:, 1:] += ty
    for row in (0, -1):
        kb = kappa[row, :]
        diag[row, :] += hy * (gamma * kb) / (kb + 0.5 * hx * gamma)
    for col in (0, -1):
        kb = kappa[:, col]
        diag[:, col] += hx * (gamma * kb) / (kb + 0.5 * hy * gamma)

    left = np.concatenate((ids[:-1, :].ravel(), ids[:, :-1].ravel()))
    right = np.concatenate((ids[1:, :].ravel(), ids[:, 1:].ravel()))
    face = -np.concatenate((tx.ravel(), ty.ravel()))
    return sparse.csr_matrix(
        (np.concatenate((diag.ravel(), face, face)),
         (np.concatenate((ids.ravel(), left, right)),
          np.concatenate((ids.ravel(), right, left)))),
        shape=(kappa.size, kappa.size),
    )


def _solve_fields(matrix, rhs):
    """Reuse one SPD preconditioner for all source right-hand sides."""
    order = reverse_cuthill_mckee(matrix, symmetric_mode=True)
    a = matrix[order, :][:, order]
    b = rhs[order, :]
    # Use the incomplete factor's lower triangle to form L D L.T; applying
    # the raw (generally nonsymmetric) ILU inverse would not be valid for CG.
    ilu = spilu(a.tocsc(), drop_tol=1e-4, fill_factor=12,
                permc_spec='NATURAL', diag_pivot_thresh=0)
    lower = ilu.L.tocsr()
    upper = lower.T.tocsr()
    d = np.abs(ilu.U.diagonal())

    def precondition(v):
        v = spsolve_triangular(lower, v, lower=True, unit_diagonal=True)
        return spsolve_triangular(upper, v / d, lower=False, unit_diagonal=True)

    precond = LinearOperator(a.shape, matvec=precondition, dtype=a.dtype)
    result = np.empty_like(b)
    direct = None
    for m, bm in enumerate(b.T):
        um, info = cg(a, bm, M=precond, rtol=1e-12, atol=0.0, maxiter=600)
        # Test the true residual as well as CG's recursively updated residual.
        if info != 0 or np.linalg.norm(a @ um - bm) > 1e-10 * np.linalg.norm(bm):
            if direct is None:
                direct = splu(a.tocsc())
            um = direct.solve(bm)
        result[order, m] = um
    return result


def _profile(s, z, y, nuisance_ridge, source_ridge):
    """Eliminate nuisance by orthogonal projection in augmented space."""
    p = z.shape[1]
    m = s.shape[1]
    design = np.vstack((z, np.sqrt(nuisance_ridge) * np.eye(p)))
    targets = np.vstack((np.column_stack((s, y)), np.zeros((p, m + y.shape[1]))))
    # An orthogonal complement avoids subtracting nearly equal Gram matrices.
    u, sv, _ = np.linalg.svd(design, full_matrices=False)
    if sv.size:
        keep = sv > np.finfo(float).eps * max(design.shape) * sv[0]
        u = u[:, keep]
        targets -= u @ (u.T @ targets)
    s_perp, y_perp = targets[:, :m], targets[:, m:]
    numer = s_perp.T @ y_perp
    denom = np.sum(s_perp**2, axis=0)[:, None]
    amplitudes = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    joint_design = np.vstack((s_perp, np.sqrt(source_ridge) * np.eye(m)))
    joint_targets = np.vstack((y_perp, np.zeros((m, y.shape[1]))))
    weights = np.linalg.lstsq(joint_design, joint_targets, rcond=None)[0]
    return amplitudes, weights


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return PDE basis fields, unpenalized single fits, and joint ridge fits."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    matrix = _robin_matrix(kappa, sigma, problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], kappa.size).T
    fields = _solve_fields(matrix, rhs).T.reshape(sources.shape)
    indices = np.asarray(problem['sensor_ij'])
    s = fields[:, indices[:, 0], indices[:, 1]].T
    amplitudes, weights = _profile(
        s, np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float), problem['nuisance_ridge'],
        problem['source_ridge'],
    )
    return {
        'basis_fields': fields,
        'single_source_profiled_amplitudes': amplitudes,
        'joint_source_weights': weights,
    }
