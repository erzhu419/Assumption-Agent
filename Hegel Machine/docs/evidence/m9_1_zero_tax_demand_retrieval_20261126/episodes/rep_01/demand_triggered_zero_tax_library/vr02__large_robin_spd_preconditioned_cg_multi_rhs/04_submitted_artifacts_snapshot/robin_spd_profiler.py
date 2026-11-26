"""Cell-centered Robin diffusion and nuisance-profiled source fits."""

import numpy as np
from scipy.linalg import qr, lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_operator(kappa, sigma, hx, hy, gamma):
    nx, ny = kappa.shape
    ids = np.arange(nx * ny).reshape(nx, ny)
    diag = hx * hy * sigma.copy()
    for edge in (0, nx - 1):
        k = kappa[edge, :]
        diag[edge, :] += hy * gamma * k / (k + 0.5 * hx * gamma)
    for edge in (0, ny - 1):
        k = kappa[:, edge]
        diag[:, edge] += hx * gamma * k / (k + 0.5 * hy * gamma)

    left = (slice(None, -1), slice(None))
    right = (slice(1, None), slice(None))
    bottom = (slice(None), slice(None, -1))
    top = (slice(None), slice(1, None))
    rows, cols, values = [], [], []
    for low, high, ratio in ((left, right, hy / hx), (bottom, top, hx / hy)):
        a, b = kappa[low], kappa[high]
        g = (2.0 * ratio) * a * b / (a + b)
        diag[low] += g
        diag[high] += g
        i, j = ids[low].ravel(), ids[high].ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-g.ravel(), -g.ravel()))
    rows.append(ids.ravel())
    cols.append(ids.ravel())
    values.append(diag.ravel())
    return coo_matrix((np.concatenate(values),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny)).tocsc()


def _profile(S, Y, Z, lam_z, lam_s):
    n, m = S.shape
    p = Z.shape[1]
    if p:
        # Appending sqrt(lam_z) I expresses the nuisance penalty as residual rows.
        # The trailing QR coordinates eliminate nuisance without forming a Gram matrix.
        augmented_Z = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        Q, R, _ = qr(augmented_Z, pivoting=True)
        d = np.abs(np.diag(R))
        rank = np.count_nonzero(d > np.finfo(float).eps * max(n + p, p) * d[0])
        complement = Q[:n, rank:].T
        S, Y = complement @ S, complement @ Y

    denom = np.einsum('ij,ij->j', S, S)[:, None]
    cross = S.T @ Y
    single = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    if lam_s:
        S = np.vstack((S, np.sqrt(lam_s) * np.eye(m)))
        Y = np.vstack((Y, np.zeros((m, Y.shape[1]))))
    joint = lstsq(S, Y)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the shared finite-volume system and fit each observation column."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    A = _robin_operator(kappa, sigma, hx, hy, problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m = sources.shape[0]
    rhs = (hx * hy) * sources.reshape(m, -1).T
    # Factor once for the entire source bank on this modest five-point grid.
    fields = splu(A).solve(rhs).T.reshape(m, nx, ny)

    ij = np.asarray(problem['sensor_ij'])
    S = fields[:, ij[:, 0], ij[:, 1]].T
    single, joint = _profile(S, np.asarray(problem['Y_obs'], dtype=float),
                             np.asarray(problem['nuisance_design'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
