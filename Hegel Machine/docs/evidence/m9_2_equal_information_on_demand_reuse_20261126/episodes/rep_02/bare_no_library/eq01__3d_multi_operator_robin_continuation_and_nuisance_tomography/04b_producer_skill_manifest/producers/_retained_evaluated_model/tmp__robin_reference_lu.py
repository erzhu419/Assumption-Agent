"""Finite-volume Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


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
    lu = _factor(assemble(theta_ref), permutation)

    def precondition(r):
        return lu.solve(r[permutation])[inverse]

    pre = LinearOperator((n, n), matvec=precondition, dtype=float)
    sensor = cells[tuple(np.asarray(study['sensor_ijk']).T)]
    stacked = np.empty((len(thetas), len(sensor), modes))
    b = vol * source.reshape(modes, n)
    scales = np.linalg.norm(b, axis=1)
    fields = np.zeros_like(b)
    for idx in np.argsort(thetas):
        a = assemble(float(thetas[idx]))
        for mode in range(modes):
            if scales[mode] == 0:
                continue
            rhs = b[mode]
            x, _ = cg(a, rhs, x0=fields[mode], M=pre, rtol=3e-11, maxiter=100)
            # Check the true residual, not only the CG recurrence.
            residual = rhs - a @ x
            if np.linalg.norm(residual) > 1e-9 * scales[mode]:
                delta, _ = cg(a, residual, M=pre, rtol=1e-7, maxiter=150)
                x += delta
                if np.linalg.norm(rhs - a @ x) > 2e-8 * scales[mode]:
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
