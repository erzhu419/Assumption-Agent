"""Cell-centered Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.linalg.blas import dtbsv
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg, splu


def _operator(problem):
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    gamma = problem['gamma_robin']

    wx = (hy / hx) * (2 * k[:-1] * k[1:] / (k[:-1] + k[1:]))
    wy = (hx / hy) * (2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]))
    area = hx * hy
    main = area * np.asarray(problem['sigma'], dtype=float)
    main[:-1] += wx
    main[1:] += wx
    main[:, :-1] += wy
    main[:, 1:] += wy
    for i in (0, -1):
        main[i] += hy * gamma * k[i] / (k[i] + gamma * hx / 2)
    for j in (0, -1):
        main[:, j] += hx * gamma * k[:, j] / (k[:, j] + gamma * hy / 2)

    # A zero at each row break prevents index-adjacent cells across the seam
    # from acquiring a spurious y-face in C-order storage.
    x = wx.ravel()
    y = np.pad(wy, ((0, 0), (0, 1))).ravel()[:-1]
    d = main.ravel()
    a = diags((-x, -y, d, -y, -x), (-ny, -1, 0, 1, ny), format='csr')
    return a, d, x, y, area


def _preconditioner(d, x, y, ny, shape):
    """Build a shared zero-fill incomplete LDLᵀ factorization."""
    piv = d.copy()
    lx = np.empty_like(x)
    ly = np.empty_like(y)
    # A five-point grid has no triangles, so retained off-diagonal factors
    # have no common-predecessor updates; only the pivots receive updates.
    for i, pivot in enumerate(piv):
        if i < y.size:
            ly[i] = -y[i] / pivot
            piv[i + 1] += y[i] * ly[i]
        if i < x.size:
            lx[i] = -x[i] / pivot
            piv[i + ny] += x[i] * lx[i]
    # Store the triangular factor once in BLAS band layout. Both the
    # forward and transposed solves then reuse it without sparse setup.
    lower = np.zeros((ny + 1, d.size), order='F')
    lower[0] = 1
    lower[1, :-1] = ly
    lower[ny, :-ny] = lx

    def apply(r):
        z = dtbsv(ny, lower, r, lower=1, diag=1)
        return dtbsv(ny, lower, z / piv, lower=1, trans=1, diag=1, overwrite_x=1)

    return LinearOperator(shape, matvec=apply, dtype=float)


def _profile(s, z, y, lam_z, lam_s):
    m, p = s.shape[1], z.shape[1]
    # In this augmented space an ordinary orthogonal projection eliminates
    # nuisance while leaving the source amplitudes unpenalized.
    design = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
    data = np.column_stack((s, y))
    data = np.vstack((data, np.zeros((p, data.shape[1]))))
    residualized = data - design @ np.linalg.lstsq(design, data, rcond=None)[0]
    sp, yp = residualized[:, :m], residualized[:, m:]
    numer = sp.T @ yp
    denom = np.einsum('ij,ij->j', sp, sp)[:, None]
    single = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    joint = np.linalg.lstsq(
        np.vstack((sp, np.sqrt(lam_s) * np.eye(m))),
        np.vstack((yp, np.zeros((m, yp.shape[1])))),
        rcond=None,
    )[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all candidate loads and fit single and joint source weights."""
    a, d, x, y, area = _operator(problem)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m, nx, ny = sources.shape
    rhs = area * sources.reshape(m, -1)
    preconditioner = _preconditioner(d, x, y, ny, a.shape)
    solutions = np.empty_like(rhs)
    direct = None
    for i, b in enumerate(rhs):
        scale = np.linalg.norm(b)
        if scale == 0:
            solutions[i] = 0
            continue
        u, info = cg(a, b, M=preconditioner, rtol=5e-12, atol=0.0, maxiter=800)
        # Check the actual stencil residual rather than trusting the
        # recursively updated Krylov residual at high contrast.
        if info < 0 or np.linalg.norm(a @ u - b) > 2e-10 * scale:
            if direct is None:
                direct = splu(a.tocsc())
            u = direct.solve(b)
        solutions[i] = u

    fields = solutions.reshape(m, nx, ny)
    sensors = np.asarray(problem['sensor_ij'])
    s = fields[:, sensors[:, 0], sensors[:, 1]].T
    single, joint = _profile(
        s, np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float),
        problem['nuisance_ridge'], problem['source_ridge'],
    )
    return {
        'basis_fields': fields,
        'single_source_profiled_amplitudes': single,
        'joint_source_weights': joint,
    }
