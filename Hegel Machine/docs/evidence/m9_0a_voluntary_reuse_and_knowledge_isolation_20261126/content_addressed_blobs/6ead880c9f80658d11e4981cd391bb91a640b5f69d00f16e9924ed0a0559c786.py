"""Conservative Robin diffusion and ridge-profiled source matching."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg


def _robin_operator(k, sigma, gamma, hx, hy):
    nx, ny = k.shape
    index = np.arange(k.size).reshape(k.shape)
    x = (hy / hx) * (2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :]))
    y = (hx / hy) * (2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]))
    diag = hx * hy * np.broadcast_to(sigma, k.shape).copy()
    diag[:-1, :] += x
    diag[1:, :] += x
    diag[:, :-1] += y
    diag[:, 1:] += y
    for i in (0, nx - 1):
        diag[i, :] += hy * gamma * k[i, :] / (k[i, :] + 0.5 * hx * gamma)
    for j in (0, ny - 1):
        diag[:, j] += hx * gamma * k[:, j] / (k[:, j] + 0.5 * hy * gamma)
    left = np.concatenate((index[:-1, :].ravel(), index[:, :-1].ravel()))
    right = np.concatenate((index[1:, :].ravel(), index[:, 1:].ravel()))
    face = np.concatenate((x.ravel(), y.ravel()))
    ids = index.ravel()
    return sparse.csr_matrix((np.r_[diag.ravel(), -face, -face],
                              (np.r_[ids, left, right], np.r_[ids, right, left])),
                             shape=(k.size, k.size))


def _aggregate(a, diagonal):
    edges = sparse.triu(a, k=1, format='coo')
    strength = np.abs(edges.data) / np.sqrt(diagonal[edges.row] * diagonal[edges.col])
    order = np.argsort(-strength, kind='stable')
    rows, cols = edges.row[order].tolist(), edges.col[order].tolist()
    strong = np.count_nonzero(strength >= 0.08)
    n = a.shape[0]
    label = [-1] * n
    count = 0

    def match(start, stop):
        nonlocal count
        for e in range(start, stop):
            i, j = rows[e], cols[e]
            if label[i] == label[j] == -1:
                label[i] = label[j] = count
                count += 1

    match(0, strong)
    # Weak edges are only needed when strong matching leaves too many singletons.
    if n - count > 0.76 * n:
        match(strong, len(rows))
    for i in range(n):
        if label[i] == -1:
            label[i] = count
            count += 1
    return sparse.csr_matrix((np.ones(n), (np.arange(n), label)), shape=(n, count))


def _galerkin_preconditioner(a):
    """Build a symmetric smoothed-aggregation V-cycle shared by every RHS."""
    levels = []
    fine_n = a.shape[0]
    while a.shape[0] > 64:
        diagonal = a.diagonal()
        # Gershgorin bound for D^-1 A (similar to the symmetric scaled A).
        bound = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / diagonal)
        relax = (4.0 / (3.0 * bound)) / diagonal
        tentative = _aggregate(a, diagonal)
        p = (tentative - (a @ tentative).multiply(relax[:, None])).tocsr()
        pt = p.T.tocsr()
        levels.append((a, relax, p, pt))
        coarse = pt @ (a @ p)
        a = ((coarse + coarse.T) * 0.5).tocsr()
        a.eliminate_zeros()
    factor = linalg.cho_factor(a.toarray(), check_finite=False)

    def cycle(level, b):
        if level == len(levels):
            return linalg.cho_solve(factor, b, check_finite=False)
        matrix, relax, p, pt = levels[level]
        x = relax * b
        x += p @ cycle(level + 1, pt @ (b - matrix @ x))
        return x + relax * (b - matrix @ x)

    return LinearOperator((fine_n, fine_n), matvec=lambda b: cycle(0, b), dtype=a.dtype)


def _solve_fields(a, sources, cell_area):
    preconditioner = _galerkin_preconditioner(a)
    rhs = cell_area * sources.reshape(sources.shape[0], -1)
    fields = np.zeros_like(rhs)
    for m, b in enumerate(rhs):
        norm = np.linalg.norm(b)
        if norm == 0:
            continue
        x, _ = cg(a, b, M=preconditioner, rtol=2e-11, atol=0.0, maxiter=450)
        r = b - a @ x
        for _ in range(3):
            if np.linalg.norm(r) <= 2e-10 * norm:
                break
            dx, _ = cg(a, r, M=preconditioner, rtol=1e-3, atol=0.0, maxiter=450)
            x += dx
            r = b - a @ x
        if not np.linalg.norm(r) <= 1e-8 * norm:
            raise RuntimeError(f'Robin CG did not meet the residual tolerance for source {m}')
        fields[m] = x
    return fields.reshape(sources.shape)


def _profile(s, z, y, lam_z, lam_s):
    # A square-root factor of I - Z (Z.T Z + lam_z I)^-1 Z.T.
    # This avoids subtracting nearly equal Gram energies in the profile.
    u, d, _ = linalg.svd(z, full_matrices=False, check_finite=False)
    if lam_z > 0:
        shrink = np.sqrt(lam_z) / np.hypot(d, np.sqrt(lam_z))
    else:
        rank = np.count_nonzero(d > np.finfo(float).eps * max(z.shape) * d[0]) if d.size else 0
        u = u[:, :rank]
        shrink = np.zeros(rank)

    def transform(v):
        coords = u.T @ v
        return np.vstack((v - u @ coords, shrink[:, None] * coords))

    design, values = transform(s), transform(y)
    energy = np.einsum('ij,ij->j', design, design)
    # Use zero for an amplitude whose field is identically unobservable.
    floor = (np.finfo(float).eps * max(z.shape) * np.linalg.norm(s, axis=0)) ** 2
    observable = energy > (0 if lam_z > 0 else floor)
    single = np.divide(design.T @ values, energy[:, None],
                       out=np.zeros((s.shape[1], y.shape[1])), where=observable[:, None])
    if lam_z == 0:
        design[:, ~observable] = 0
    if lam_s > 0:
        design = np.vstack((design, np.sqrt(lam_s) * np.eye(s.shape[1])))
        values = np.vstack((values, np.zeros((s.shape[1], y.shape[1]))))
    joint = linalg.lstsq(design, values, check_finite=False)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the finite-volume equations and profile nuisance for each experiment."""
    k = np.asarray(problem['kappa'], dtype=float)
    hx = problem['Lx'] / k.shape[0]
    hy = problem['Ly'] / k.shape[1]
    a = _robin_operator(k, problem['sigma'], problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = _solve_fields(a, sources, hx * hy)
    i, j = np.asarray(problem['sensor_ij']).T
    single, joint = _profile(fields[:, i, j].T,
                             np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
