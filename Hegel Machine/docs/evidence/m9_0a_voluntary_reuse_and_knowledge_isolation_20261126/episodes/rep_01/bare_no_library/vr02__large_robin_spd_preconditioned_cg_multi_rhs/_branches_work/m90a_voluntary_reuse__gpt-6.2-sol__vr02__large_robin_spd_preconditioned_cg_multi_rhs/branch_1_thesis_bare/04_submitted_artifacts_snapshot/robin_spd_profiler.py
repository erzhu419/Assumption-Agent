"""Cell-centered Robin finite volumes and nuisance-profiled source fitting."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, splu


def _assemble(problem):
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    robin = problem['gamma_robin']
    ids = np.arange(k.size).reshape(k.shape)

    kx = (2 * hy / hx) * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    ky = (2 * hx / hy) * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    diagonal = area * np.asarray(problem['sigma'], dtype=float)
    diagonal[:-1, :] += kx
    diagonal[1:, :] += kx
    diagonal[:, :-1] += ky
    diagonal[:, 1:] += ky
    for i in (0, -1):
        diagonal[i, :] += hy * robin * k[i, :] / (k[i, :] + 0.5 * hx * robin)
    for j in (0, -1):
        diagonal[:, j] += hx * robin * k[:, j] / (k[:, j] + 0.5 * hy * robin)

    left = np.concatenate((ids[:-1, :].ravel(), ids[:, :-1].ravel()))
    right = np.concatenate((ids[1:, :].ravel(), ids[:, 1:].ravel()))
    edge = -np.concatenate((kx.ravel(), ky.ravel()))
    flat = ids.ravel()
    matrix = coo_matrix((np.concatenate((diagonal.ravel(), edge, edge)),
                         (np.concatenate((flat, left, right)),
                          np.concatenate((flat, right, left)))),
                        shape=(k.size, k.size)).tocsr()
    return matrix, area, diagonal, kx, ky, ids, left, right, edge


def _preconditioner(diagonal, kx, ky, ids, left, right, edge):
    # IC(0) retains just the west and south entries of each lower row.
    # The five-point graph has no triangles, so retained off-diagonals
    # share no earlier pivot; only the diagonal update has a sum.
    nx, ny = diagonal.shape
    pivot = np.empty_like(diagonal)
    for i in range(nx):
        for j in range(ny):
            value = diagonal[i, j]
            if i:
                value -= (kx[i - 1, j] / pivot[i - 1, j]) ** 2
            if j:
                value -= (ky[i, j - 1] / pivot[i, j - 1]) ** 2
            pivot[i, j] = np.sqrt(value)
    d = pivot.ravel()
    lower = coo_matrix((np.concatenate((d, edge / d[left])),
                        (np.concatenate((ids.ravel(), right)),
                         np.concatenate((ids.ravel(), left)))),
                       shape=(d.size, d.size)).tocsc()
    # Factor the already triangular L once to reuse sparse substitutions.
    triangular = splu(lower, permc_spec='NATURAL', diag_pivot_thresh=0)

    def apply(r):
        return triangular.solve(triangular.solve(r), trans='T')

    return LinearOperator(lower.shape, matvec=apply, dtype=float)


def _profile(s, z, y, lam_z, lam_s):
    # Encode the nuisance penalty as additional observations with zero data.
    p = z.shape[1]
    z_aug = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
    s_aug = np.pad(s, ((0, p), (0, 0)))
    y_aug = np.pad(y, ((0, p), (0, 0)))
    if p:
        u, singular, _ = np.linalg.svd(z_aug, full_matrices=False)
        cutoff = np.finfo(float).eps * max(z_aug.shape) * singular[0]
        q = u[:, singular > cutoff]
        s_aug -= q @ (q.T @ s_aug)
        y_aug -= q @ (q.T @ y_aug)

    numerator = s_aug.T @ y_aug
    denominator = np.sum(s_aug * s_aug, axis=0)
    # An unidentifiable single mode has no unique amplitude; choose zero.
    floor = (np.finfo(float).eps * max(z_aug.shape) * np.linalg.norm(s, axis=0)) ** 2
    single = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator),
                       where=denominator[:, None] > floor[:, None])

    m = s.shape[1]
    joint = np.linalg.lstsq(np.vstack((s_aug, np.sqrt(lam_s) * np.eye(m))),
                            np.pad(y_aug, ((0, m), (0, 0))), rcond=None)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all source modes and fit the sensor experiments."""
    matrix, area, *stencil = _assemble(problem)
    preconditioner = _preconditioner(*stencil)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = np.empty_like(sources)
    for m, source in enumerate(sources):
        rhs = area * source.ravel()
        field, info = cg(matrix, rhs, M=preconditioner, rtol=1e-12, atol=0)
        if info or np.linalg.norm(matrix @ field - rhs) > 1e-8 * np.linalg.norm(rhs):
            raise RuntimeError(f'PCG did not converge for source {m}')
        fields[m] = field.reshape(source.shape)

    i, j = np.asarray(problem['sensor_ij']).T
    single, joint = _profile(fields[:, i, j].T,
                             np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
