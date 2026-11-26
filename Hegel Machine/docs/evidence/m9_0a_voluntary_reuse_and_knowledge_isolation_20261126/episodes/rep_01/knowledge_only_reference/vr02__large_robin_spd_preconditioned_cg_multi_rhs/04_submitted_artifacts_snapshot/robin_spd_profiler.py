"""Conservative Robin diffusion-reaction solves and ridge-profiled source fits."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg


def _assemble(k, sigma, gamma, hx, hy):
    """Assemble one shared conductance per interior face in C-order."""
    nodes = np.arange(k.size).reshape(k.shape)
    ax, bx = k[:-1, :], k[1:, :]
    ay, by = k[:, :-1], k[:, 1:]
    wx = (hy / hx) * (2 * ax * bx / (ax + bx))
    wy = (hx / hy) * (2 * ay * by / (ay + by))
    diag = np.array(sigma * (hx * hy), dtype=float, copy=True)
    diag[:-1, :] += wx
    diag[1:, :] += wx
    diag[:, :-1] += wy
    diag[:, 1:] += wy
    for i in (0, -1):
        diag[i, :] += hy * gamma * k[i, :] / (k[i, :] + gamma * hx / 2)
    for j in (0, -1):
        diag[:, j] += hx * gamma * k[:, j] / (k[:, j] + gamma * hy / 2)
    u = np.r_[nodes[:-1, :].ravel(), nodes[:, :-1].ravel()]
    v = np.r_[nodes[1:, :].ravel(), nodes[:, 1:].ravel()]
    w = np.r_[wx.ravel(), wy.ravel()]
    return sparse.csr_matrix((np.r_[diag.ravel(), -w, -w],
                              (np.r_[nodes.ravel(), u, v],
                               np.r_[nodes.ravel(), v, u])), shape=(k.size, k.size))


def _aggregate(a, diag):
    """Match strong normalized edges, then attach unmatched neighbors."""
    edges = sparse.triu(a, k=1, format='coo')
    strength = np.abs(edges.data) / np.sqrt(diag[edges.row] * diag[edges.col])
    order = np.argsort(-strength, kind='stable')
    n = a.shape[0]
    for cutoff in (0.08, 0.0):
        selected = order[strength[order] > cutoff]
        neighbors = list(zip(edges.row[selected], edges.col[selected]))
        groups = np.full(n, -1, dtype=int)
        count = 0
        for i, j in neighbors:
            if groups[i] < 0 and groups[j] < 0:
                groups[i] = groups[j] = count
                count += 1
        for i, j in neighbors:
            if groups[i] < 0 <= groups[j]:
                groups[i] = groups[j]
            elif groups[j] < 0 <= groups[i]:
                groups[j] = groups[i]
        alone = groups < 0
        groups[alone] = np.arange(count, count + alone.sum())
        count += alone.sum()
        if count <= 0.76 * n:
            break
    return sparse.csr_matrix((np.ones(n), (np.arange(n), groups)), shape=(n, count))


def _preconditioner(a):
    """Build a symmetric smoothed-aggregation Galerkin V-cycle once."""
    levels = []
    while a.shape[0] > 64:
        d = a.diagonal()
        bound = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / d)
        step = (4 / (3 * bound)) / d
        p = _aggregate(a, d)
        if p.shape[1] == a.shape[0]:
            break
        p = (p - (a @ p).multiply(step[:, None])).tocsr()
        pt = p.T.tocsr()
        levels.append((a, step, p, pt))
        coarse = pt @ (a @ p)
        a = ((coarse + coarse.T) * 0.5).tocsr()
    if a.shape[0] <= 64:
        factor = linalg.cho_factor(a.toarray(), check_finite=False)

        def bottom(b):
            return linalg.cho_solve(factor, b, check_finite=False)
    else:
        d = a.diagonal()

        def bottom(b):
            return b / d

    def cycle(level, b):
        if level == len(levels):
            return bottom(b)
        mat, step, p, pt = levels[level]
        x = step * b
        x += step * (b - mat @ x)
        x += p @ cycle(level + 1, pt @ (b - mat @ x))
        x += step * (b - mat @ x)
        x += step * (b - mat @ x)
        return x

    n = levels[0][0].shape[0] if levels else a.shape[0]
    return LinearOperator((n, n), matvec=lambda b: cycle(0, b), dtype=float)


def _solve(a, preconditioner, b):
    norm = np.linalg.norm(b)
    if norm == 0:
        return np.zeros_like(b)
    solution = np.zeros_like(b)
    residual = b
    for _ in range(4):
        correction, _ = cg(a, residual, M=preconditioner, rtol=3e-13,
                           atol=0.0, maxiter=450)
        solution += correction
        residual = b - a @ solution
        relative = np.linalg.norm(residual) / norm
        if relative <= 3e-12:
            return solution
    if not relative <= 1e-8:
        raise RuntimeError(f'Robin solve failed residual tolerance: {relative:.3g}')
    return solution


def _profile(s, z, y, lam_z, lam_s):
    """Project in augmented least-squares coordinates; avoid Gram subtraction."""
    p = z.shape[1]
    if p:
        z_aug = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
        u, singular, _ = linalg.svd(z_aug, full_matrices=False, check_finite=False)
        if lam_z == 0:
            u = u[:, singular > np.finfo(float).eps * max(z.shape) * singular[0]]
        s_aug = np.vstack((s, np.zeros((p, s.shape[1]))))
        y_aug = np.vstack((y, np.zeros((p, y.shape[1]))))
        sr = s_aug - u @ (u.T @ s_aug)
        yr = y_aug - u @ (u.T @ y_aug)
    else:
        sr, yr = s, y
    # A candidate annihilated by the projection has no identifiable amplitude.
    magnitude = np.linalg.norm(sr, axis=0)
    cutoff = np.finfo(float).eps * max(s.shape) * np.linalg.norm(s, axis=0)
    good = magnitude > cutoff
    if lam_z > 0:
        good = magnitude > 0
    singles = np.zeros((s.shape[1], y.shape[1]))
    singles[good] = ((sr[:, good] / magnitude[good]).T @ yr) / magnitude[good, None]
    if lam_s:
        m = s.shape[1]
        sr = np.vstack((sr, np.sqrt(lam_s) * np.eye(m)))
        yr = np.vstack((yr, np.zeros((m, y.shape[1]))))
    else:
        sr = sr.copy()
        sr[:, ~good] = 0.0
    joint = linalg.lstsq(sr, yr, check_finite=False)[0]
    return singles, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return the PDE basis, independent single-source fits, and joint weights."""
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    a = _assemble(k, np.asarray(problem['sigma']), problem['gamma_robin'], hx, hy)
    preconditioner = _preconditioner(a)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = np.empty_like(sources)
    for field, source in zip(fields, sources):
        field[...] = _solve(a, preconditioner, source.ravel() * (hx * hy)).reshape(k.shape)
    sensors = np.asarray(problem['sensor_ij'])
    s = fields[:, sensors[:, 0], sensors[:, 1]].T
    single, joint = _profile(s, np.asarray(problem['nuisance_design']),
                             np.asarray(problem['Y_obs']), problem['nuisance_ridge'],
                             problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
