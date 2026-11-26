"""Cell-centered Robin diffusion and nuisance-profiled source fits."""

import numpy as np
from scipy.linalg import qr
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg, splu


def _operator(k, sigma, gamma, hx, hy):
    nx, ny = k.shape
    x = (hy / hx) * (2 * k[:-1] * k[1:] / (k[:-1] + k[1:]))
    y = (hx / hy) * (2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]))
    d = np.broadcast_to(np.asarray(sigma, dtype=float), k.shape).copy() * (hx * hy)
    d[:-1] += x
    d[1:] += x
    d[:, :-1] += y
    d[:, 1:] += y
    for i in (0, nx - 1):
        d[i] += hy * gamma * k[i] / (k[i] + 0.5 * hx * gamma)
    for j in (0, ny - 1):
        d[:, j] += hx * gamma * k[:, j] / (k[:, j] + 0.5 * hy * gamma)
    y_band = np.zeros((nx, ny))
    y_band[:, :-1] = -y
    xb, yb = -x.ravel(), y_band.ravel()[:-1]
    A = diags([xb, yb, d.ravel(), yb, xb], [-ny, -1, 0, 1, ny], format='csr')
    return A, d, x, y


def _incomplete_cholesky(d, x, y):
    """IC(0) on the grid graph, evaluated in dependency wavefronts."""
    nx, ny = d.shape
    p = d.copy()
    lx, ly = np.zeros_like(d), np.zeros_like(d)
    for t in range(nx + ny - 1):
        i = np.arange(max(0, t - ny + 1), min(nx - 1, t) + 1)
        j = t - i
        a, b = i > 0, j > 0
        lx[i[a], j[a]] = -x[i[a] - 1, j[a]] / p[i[a] - 1, j[a]]
        ly[i[b], j[b]] = -y[i[b], j[b] - 1] / p[i[b], j[b] - 1]
        p[i, j] = np.sqrt(d[i, j] - lx[i, j] ** 2 - ly[i, j] ** 2)
    # Adjacent grid vertices have no shared earlier neighbor, so the
    # stored off-diagonals require no additional dot-product terms.
    L = diags([lx.ravel()[ny:], ly.ravel()[1:], p.ravel()], [-ny, -1, 0], format='csc')
    # Factor the already triangular IC(0) factor once so the repeated
    # forward and backward substitutions run in SuperLU's compiled solve.
    triangular = splu(L, permc_spec='NATURAL', diag_pivot_thresh=0)

    def apply(r):
        return triangular.solve(triangular.solve(r), trans='T')

    return LinearOperator(L.shape, matvec=apply, dtype=float)


def _solve_fields(A, preconditioner, rhs):
    fields = np.empty_like(rhs)
    direct = None
    for m, b in enumerate(rhs):
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            fields[m] = 0
            continue
        u, info = cg(A, b, M=preconditioner, rtol=2e-13, maxiter=500)
        # Check the actual residual: CG's recursively updated residual can drift.
        if info != 0 or np.linalg.norm(b - A @ u) > 5e-11 * norm_b:
            if direct is None:
                direct = splu(A.tocsc())
            u = direct.solve(b)
        fields[m] = u
    return fields


def _profile(S, Z, Y, lam_z, lam_s):
    p = Z.shape[1]
    # Rotate the augmented nuisance system and retain its orthogonal complement.
    # Its top rows are the observations; the bottom rows encode nuisance ridge.
    if p:
        Za = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        data = np.pad(np.column_stack((S, Y)), ((0, p), (0, 0)))
        Q, R, _ = qr(Za, mode='full', pivoting=True)
        if lam_z > 0:
            rank = p
        else:
            diagonal = np.abs(np.diag(R))
            rank = np.count_nonzero(diagonal > diagonal.max() * max(Za.shape) * np.finfo(float).eps)
        profiled = Q[:, rank:].T @ data
        S, Y = np.split(profiled, [S.shape[1]], axis=1)
    denominator = np.einsum('im,im->m', S, S)[:, None]
    numerator = S.T @ Y
    single = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    m = S.shape[1]
    joint = np.linalg.lstsq(np.vstack((S, np.sqrt(lam_s) * np.eye(m))),
                            np.pad(Y, ((0, m), (0, 0))), rcond=None)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return source response fields and the two profiled sensor fits."""
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    A, d, x, y = _operator(k, problem['sigma'], problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny)
    fields = _solve_fields(A, _incomplete_cholesky(d, x, y), rhs).reshape(sources.shape)
    ij = np.asarray(problem['sensor_ij'])
    S = fields[:, ij[:, 0], ij[:, 1]].T
    single, joint = _profile(S, np.asarray(problem['nuisance_design'], dtype=float),
                            np.asarray(problem['Y_obs'], dtype=float),
                            problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
