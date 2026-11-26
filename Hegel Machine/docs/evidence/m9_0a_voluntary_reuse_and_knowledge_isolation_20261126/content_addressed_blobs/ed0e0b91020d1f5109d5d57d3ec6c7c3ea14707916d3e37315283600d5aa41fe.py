"""Cell-centered Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu


def _operator(problem):
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    g = problem['gamma_robin']

    kl, kr = k[:-1, :], k[1:, :]
    tx = (hy / hx) * (2 * kl * kr / (kl + kr))
    kl, kr = k[:, :-1], k[:, 1:]
    ty = (hx / hy) * (2 * kl * kr / (kl + kr))

    diag = area * np.asarray(problem['sigma'], dtype=float) + np.zeros_like(k)
    diag[:-1, :] += tx
    diag[1:, :] += tx
    diag[:, :-1] += ty
    diag[:, 1:] += ty
    for i in (0, nx - 1):
        diag[i, :] += hy * (g * k[i, :]) / (g * 0.5 * hx + k[i, :])
    for j in (0, ny - 1):
        diag[:, j] += hx * (g * k[:, j]) / (g * 0.5 * hy + k[:, j])

    grid = np.arange(k.size).reshape(k.shape)
    row = np.r_[grid[1:, :].ravel(), grid[:, 1:].ravel()]
    col = np.r_[grid[:-1, :].ravel(), grid[:, :-1].ravel()]
    off = -np.r_[tx.ravel(), ty.ravel()]
    flat = grid.ravel()
    A = coo_matrix((np.r_[diag.ravel(), off, off],
                    (np.r_[flat, row, col], np.r_[flat, col, row])),
                   shape=(k.size, k.size)).tocsr()

    # Zero-fill incomplete LDL^T. The grid has no triangles, so retained
    # off-diagonals need only be divided by their corresponding pivot.
    pivots = diag.copy()
    for i in range(nx):
        for j in range(ny):
            if i:
                pivots[i, j] -= tx[i - 1, j] ** 2 / pivots[i - 1, j]
            if j:
                pivots[i, j] -= ty[i, j - 1] ** 2 / pivots[i, j - 1]
    d = pivots.ravel()
    lower = coo_matrix((off / d[col], (row, col)), shape=A.shape).tocsr()
    return A, area, lower, d


def _solve_fields(A, B, lower, d):
    """Independent CG recurrences, sharing matrix and triangular solves."""
    # Factor the unit triangular matrix once to reuse compiled sparse
    # forward and transpose substitutions for every CG iteration.
    factor = splu(lower.tocsc() + eye(A.shape[0], format='csc'),
                  permc_spec='NATURAL', diag_pivot_thresh=0)

    def precondition(r):
        return factor.solve(factor.solve(r) / d[:, None], trans='T')

    X = np.zeros_like(B)
    targets = 1e-11 * np.sqrt(np.sum(B * B, axis=0))
    active = np.flatnonzero(targets > 0)
    # Recompute the true residual whenever the recurrence reports convergence.
    # If round-off caused a premature stop, restart only that column.
    while active.size:
        x = X[:, active]
        r = B[:, active] - A @ x
        z = precondition(r)
        p = z.copy()
        rz = np.einsum('ij,ij->j', r, z)
        restart = []
        for _ in range(A.shape[0]):
            ap = A @ p
            a = rz / np.einsum('ij,ij->j', p, ap)
            x += p * a
            r -= ap * a
            done = np.sqrt(np.sum(r * r, axis=0)) <= targets[active]
            if np.any(done):
                ids = active[done]
                X[:, ids] = x[:, done]
                actual = B[:, ids] - A @ X[:, ids]
                # Accept well inside the specified tolerance even if the
                # recursive residual has fallen below the precision floor.
                restart.extend(ids[np.sqrt(np.sum(actual * actual, axis=0))
                                   > 1e-10 * np.sqrt(np.sum(B[:, ids] ** 2, axis=0))])
                keep = ~done
                if not np.any(keep):
                    break
                active, x, r, p, rz = active[keep], x[:, keep], r[:, keep], p[:, keep], rz[keep]
            z = precondition(r)
            next_rz = np.einsum('ij,ij->j', r, z)
            p = z + p * (next_rz / rz)
            rz = next_rz
        else:
            raise RuntimeError('Preconditioned CG did not converge')
        active = np.asarray(restart, dtype=int)
    return X


def _profile(S, Z, Y, lam_z, lam_s):
    # Augment the data term so nuisance ridge is an ordinary least-squares
    # penalty. Orthogonal residualization applies the same projection to S/Y.
    m, q, p = S.shape[1], Y.shape[1], Z.shape[1]
    data = np.vstack((np.column_stack((S, Y)), np.zeros((p, m + q))))
    if p:
        nuisance = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        data -= nuisance @ np.linalg.lstsq(nuisance, data, rcond=None)[0]
    Sr, Yr = data[:, :m], data[:, m:]

    num = Sr.T @ Yr
    den = np.einsum('ij,ij->j', Sr, Sr)[:, None]
    single = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    joint_design = np.vstack((Sr, np.sqrt(lam_s) * np.eye(m)))
    joint_data = np.vstack((Yr, np.zeros((m, q))))
    joint = np.linalg.lstsq(joint_design, joint_data, rcond=None)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the finite-volume PDE for each candidate, then fit sensor data."""
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    A, area, lower, d = _operator(problem)
    rhs = area * sources.reshape(sources.shape[0], A.shape[0]).T
    fields = _solve_fields(A, rhs, lower, d).T.reshape(sources.shape)
    sensors = np.asarray(problem['sensor_ij'])
    S = fields[:, sensors[:, 0], sensors[:, 1]].T
    single, joint = _profile(S, np.asarray(problem['nuisance_design']),
                            np.asarray(problem['Y_obs']), problem['nuisance_ridge'],
                            problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
