"""Finite-volume Robin continuation and nuisance-profiled tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, splu


def _assembler(shape, spacing, gamma):
    ids = np.arange(np.prod(shape)).reshape(shape)
    volume = np.prod(spacing)
    faces = []
    walls = []
    for axis, h in enumerate(spacing):
        grid = ids.swapaxes(0, axis)
        faces.append((grid[:-1].ravel(), grid[1:].ravel(), volume / h**2))
        walls.append((grid[[0, -1]].ravel(), volume / h, gamma * h / 2))
    i = np.concatenate([f[0] for f in faces])
    j = np.concatenate([f[1] for f in faces])
    d = ids.ravel()
    rows = np.concatenate((d, i, j))
    cols = np.concatenate((d, j, i))

    def assemble(kappa, sigma):
        k = kappa.ravel()
        w = np.concatenate([scale * (2 * k[a] * k[b] / (k[a] + k[b]))
                            for a, b, scale in faces])
        diag = volume * sigma.ravel() + np.bincount(i, w, len(d)) + np.bincount(j, w, len(d))
        for wall, area, half in walls:
            np.add.at(diag, wall, area * gamma * k[wall] / (half + k[wall]))
        return sparse.csr_matrix((np.concatenate((diag, -w, -w)), (rows, cols)),
                                 shape=(len(d), len(d)))

    return assemble, volume


def _profile(S, Z, Y, lam_z, lam_s):
    # The residual maker for [Z; sqrt(lam_z) I] is an orthogonal
    # projector. Apply it to [S Y; 0] and take inner products without
    # subtracting nearly equal Gram matrices.
    U, t, _ = linalg.svd(Z, full_matrices=False)
    s, y = U.T @ S, U.T @ Y
    ratio = np.sqrt(lam_z) / np.hypot(t, np.sqrt(lam_z))
    tail_s = ratio[:, None] * s
    tail_y = ratio[:, None] * y
    perp_s = S - U @ s
    perp_y = Y - U @ y
    gram = perp_s.T @ perp_s + tail_s.T @ tail_s
    cross = perp_s.T @ perp_y + tail_s.T @ tail_y
    energy = np.diag(gram)[:, None]
    amplitudes = np.divide(cross, energy, out=np.zeros_like(cross), where=energy > 0)
    joint = linalg.solve(gram + lam_s * np.eye(S.shape[1]), cross, assume_a='pos')
    return amplitudes, joint


def _dot(x, y):
    # A serial reduction avoids BLAS thread overhead on these short vectors.
    return np.einsum('i,i->', x, y)


def _pcg(A, b, M, guess, tolerance, maxiter):
    x = np.zeros_like(b) if guess is None else guess.copy()
    r = b.copy() if guess is None else b - A @ x
    threshold = tolerance**2
    if _dot(r, r) <= threshold:
        return x
    p = M @ r
    rho = _dot(r, p)
    for _ in range(maxiter):
        Ap = A @ p
        alpha = rho / _dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if _dot(r, r) <= threshold:
            break
        z = M @ r
        next_rho = _dot(r, z)
        p *= next_rho / rho
        p += z
        rho = next_rho
    return x


def _iterate(A, b, preconditioner, guess):
    # Independently check the true residual; recursive CG residuals can
    # drift on high-contrast conductivity fields.
    scale = np.sqrt(_dot(b, b))
    if not scale:
        return np.zeros_like(b)
    x = _pcg(A, b, preconditioner, guess, 2e-10 * scale, 220)
    r = b - A @ x
    if np.sqrt(_dot(r, r)) > 8e-9 * scale:
        x += _pcg(A, r, preconditioner, None, 2e-10 * scale, 300)
        r = b - A @ x
        if np.sqrt(_dot(r, r)) > 3e-8 * scale:
            # Refine the current field rather than replacing it with an
            # unchecked direct solve. Extended-precision residuals avoid
            # cancellation between large diffusion terms and weak losses.
            factor = _factor(A)
            wide_A = A.astype(np.longdouble)
            wide_b = b.astype(np.longdouble)
            trial = x.copy()
            best_error = _dot(r, r)
            for _ in range(2):
                residual = np.asarray(wide_b - wide_A @ trial, dtype=float)
                trial += factor.solve(residual)
                r = b - A @ trial
                error = _dot(r, r)
                if error < best_error:
                    x = trial.copy()
                    best_error = error
    return x


def _factor(A):
    return splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                options={'SymmetricMode': True})


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve the parametric Robin family and return the four tomography arrays."""
    kb, k1, k2, sb, s1 = (np.asarray(study[name], dtype=float) for name in
                           ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    shape = kb.shape
    thetas = np.asarray(study['thetas'], dtype=float)
    spacing = np.array([study['Lx'], study['Ly'], study['Lz']]) / shape
    assemble, volume = _assembler(shape, spacing, study['gamma_robin'])

    def operator(theta):
        return assemble(kb + theta * k1 + theta**2 * k2, sb + theta * s1)

    # Positive kappa polynomial terms bound the conductance ratios on
    # either side of this geometric center, including on Robin faces.
    reference = np.sqrt(thetas.min() * thetas.max())
    factor = _factor(operator(reference))
    preconditioner = LinearOperator(factor.shape, matvec=factor.solve, dtype=float)
    rhs = (volume * np.asarray(study['source_modes'], dtype=float)).reshape(-1, kb.size)
    sensor = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, shape)
    S = np.empty((len(thetas), len(sensor), len(rhs)))
    terminal = None
    guess = [None] * len(rhs)
    for k in np.argsort(thetas):
        A = operator(thetas[k])
        fields = np.array([_iterate(A, b, preconditioner, x) for b, x in zip(rhs, guess)])
        S[k] = fields[:, sensor].T
        guess = fields
        if k == len(thetas) - 1:
            terminal = fields.reshape((-1,) + shape)
    S = S.reshape(-1, len(rhs))
    amplitudes, joint = _profile(S, np.asarray(study['Z_nuisance'], dtype=float),
                                 np.asarray(study['Y_obs'], dtype=float),
                                 study['nuisance_ridge'], study['source_ridge'])
    return dict(terminal_fields=terminal, continuation_sensor_predictions=S,
                profiled_amplitudes=amplitudes, joint_source_weights=joint)
