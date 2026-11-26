"""Finite-volume Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu


def _dissection(cells):
    """Order interiors before their planar separator to limit 3D fill."""
    axis = int(np.argmax(cells.shape))
    if cells.shape[axis] <= 3:
        return cells.ravel()
    planes = np.moveaxis(cells, axis, 0)
    middle = len(planes) // 2
    return np.concatenate((_dissection(planes[:middle]),
                           _dissection(planes[middle+1:]),
                           _dissection(planes[middle:middle+1])))


def _operator_family(study, cells):
    h = np.array([study['Lx'], study['Ly'], study['Lz']], dtype=float) / cells.shape
    vol = float(h.prod())
    n = cells.size
    faces, walls = [], []
    for axis in range(3):
        planes = np.moveaxis(cells, axis, 0)
        area = vol / h[axis]
        faces.append((planes[:-1].ravel(), planes[1:].ravel(), area / h[axis]))
        walls.append((planes[0].ravel(), area, h[axis] / 2))
        walls.append((planes[-1].ravel(), area, h[axis] / 2))
    left = np.concatenate([f[0] for f in faces])
    right = np.concatenate([f[1] for f in faces])
    cell_ids = cells.ravel()
    row = np.r_[cell_ids, left, right]
    col = np.r_[cell_ids, right, left]
    order = np.lexsort((col, row))
    indptr = np.r_[0, np.cumsum(np.bincount(row, minlength=n))].astype(np.int32)
    indices = col[order].astype(np.int32)
    k0, k1, k2, s0, s1 = (np.asarray(study[k], dtype=float).ravel() for k in
                           ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    gamma = float(study['gamma_robin'])

    def assemble(theta):
        k = k0 + theta * k1 + theta**2 * k2
        w = np.concatenate([scale * (2 * k[a] * k[b] / (k[a] + k[b]))
                            for a, b, scale in faces])
        diagonal = vol * (s0 + theta * s1)
        diagonal += np.bincount(left, weights=w, minlength=n)
        diagonal += np.bincount(right, weights=w, minlength=n)
        for ids, area, half in walls:
            diagonal[ids] += area * gamma * k[ids] / (gamma * half + k[ids])
        values = np.r_[diagonal, -w, -w][order]
        return sparse.csr_matrix((values, indices, indptr), shape=(n, n))

    return assemble, vol


def _factor(a, permutation):
    return splu(a[permutation, :][:, permutation].tocsc(), permc_spec='NATURAL',
                diag_pivot_thresh=0, options={'SymmetricMode': True})


def _multigrid(a, cells):
    """Symmetric multigrid cycle with smoothed aggregate coarse bases."""
    shape = tuple((length + 1) // 2 for length in cells.shape)
    coarse = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    membership = coarse[np.ix_(*(np.arange(length) // 2 for length in cells.shape))].ravel()
    tentative = sparse.csr_matrix((np.ones(a.shape[0]), (cells.ravel(), membership)),
                                  shape=(a.shape[0], coarse.size))
    diagonal = a.diagonal()
    # The Galerkin matrix can have positive off-diagonals, unlike the FV level.
    upper = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / diagonal)
    damping = (4 / (3 * upper)) / diagonal
    prolong = (tentative - (a @ tentative).multiply(damping[:, None])).tocsr()
    prolong.eliminate_zeros()
    restrict = prolong.T.tocsr()
    coarse_a = (restrict @ (a @ prolong)).tocsr()
    if coarse.size > 400:
        coarse_solve = _multigrid(coarse_a, coarse)
    else:
        permutation = _dissection(coarse)
        inverse = np.argsort(permutation)
        lu = _factor(coarse_a, permutation)

        def coarse_solve(rhs):
            return lu.solve(rhs[permutation])[inverse]

    def cycle(r):
        x = damping * r
        x += damping * (r - a @ x)
        rc = restrict @ (r - a @ x)
        x += prolong @ coarse_solve(rc)
        x += damping * (r - a @ x)
        x += damping * (r - a @ x)
        return x

    return cycle


def _dot(x, y):
    # Short reductions in the Krylov loop should not launch BLAS threads.
    return np.einsum('i,i->', x, y)


def _norm(x):
    return np.sqrt(_dot(x, x))


def cg(a, rhs, x0=None, M=None, rtol=1e-10, maxiter=100, callback=None):
    x = np.zeros_like(rhs) if x0 is None else x0.copy()
    r = rhs - a @ x
    target = rtol**2 * _dot(rhs, rhs)
    previous = None
    for i in range(maxiter):
        if _dot(r, r) <= target:
            return x, 0
        z = M(r)
        rz = _dot(r, z)
        if previous is None:
            direction = z
        else:
            direction *= rz / previous
            direction += z
        product = a @ direction
        step = rz / _dot(direction, product)
        x += step * direction
        r -= step * product
        previous = rz
        if callback is not None:
            callback(x)
    return x, maxiter


def _profile(study, predictors):
    z = np.asarray(study['Z_nuisance'], dtype=float)
    y = np.asarray(study['Y_obs'], dtype=float)
    modes = predictors.shape[1]
    n, q = z.shape
    # Orthogonally project in the augmented space. Its residual Gram matrix
    # equals X.T @ P_perp @ X, not X.T @ P_perp**2 @ X.
    design = np.vstack((z, np.sqrt(study['nuisance_ridge']) * np.eye(q)))
    basis = linalg.qr(design, mode='economic', check_finite=False)[0]
    data = np.zeros((n + q, modes + y.shape[1]))
    data[:n, :modes], data[:n, modes:] = predictors, y
    data -= basis @ (basis.T @ data)
    signal, obs = data[:, :modes], data[:, modes:]
    gram = signal.T @ signal
    cross = signal.T @ obs
    amplitudes = cross / np.diag(gram)[:, None]
    # Solve the augmented joint least-squares problem without squaring the
    # condition number of the source design.
    joint = linalg.lstsq(np.vstack((signal, np.sqrt(study['source_ridge']) * np.eye(modes))),
                         np.vstack((obs, np.zeros((modes, y.shape[1])))),
                         check_finite=False)[0]
    return amplitudes, joint


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve all continuation states and profile nuisance amplitudes."""
    source = np.asarray(study['source_modes'], dtype=float)
    modes, *shape = source.shape
    cells = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    n = cells.size
    assemble, vol = _operator_family(study, cells)
    thetas = np.asarray(study['thetas'], dtype=float)
    permutation = _dissection(cells)
    inverse = np.argsort(permutation)
    theta_ref = float(np.sqrt(thetas.min() * thetas.max()))
    pre = _multigrid(assemble(theta_ref), cells)
    sensor = cells[tuple(np.asarray(study['sensor_ijk']).T)]
    stacked = np.empty((len(thetas), len(sensor), modes))
    b = vol * source.reshape(modes, n)
    scales = np.linalg.norm(b, axis=1)
    fields = np.zeros_like(b)
    for idx in np.argsort(thetas):
        a = assemble(float(thetas[idx]))
        exact = None
        for mode in range(modes):
            if scales[mode] == 0:
                continue
            rhs = b[mode]
            x, _ = cg(a, rhs, x0=fields[mode], M=pre, rtol=3e-11, maxiter=100)
            # Check the true residual, not only the CG recurrence.
            residual = rhs - a @ x
            if _norm(residual) > 1e-9 * scales[mode]:
                delta, _ = cg(a, residual, M=pre, rtol=1e-7, maxiter=150)
                x += delta
                if _norm(rhs - a @ x) > 2e-8 * scales[mode]:
                    if exact is None:
                        exact = _factor(a, permutation)
                    x = exact.solve(rhs[permutation])[inverse]
            fields[mode] = x
        stacked[idx] = fields[:, sensor].T
        if idx == len(thetas) - 1:
            terminal = fields.reshape(source.shape).copy()
    stacked = stacked.reshape(-1, modes)
    amplitudes, joint = _profile(study, stacked)
    return dict(terminal_fields=terminal, continuation_sensor_predictions=stacked,
                profiled_amplitudes=amplitudes, joint_source_weights=joint)
