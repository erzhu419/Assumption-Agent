"""Cell-centered Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits


def _operators(shape, spacing, gamma):
    """Precompute C-order face indices and return the operator assembler."""
    ids = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    vol = np.prod(spacing)
    faces = []
    for axis, h in enumerate(spacing):
        grid = np.moveaxis(ids, axis, 0)
        faces.append((grid[:-1].ravel(), grid[1:].ravel(),
                      np.concatenate((grid[0].ravel(), grid[-1].ravel())), h))
    i = np.concatenate([f[0] for f in faces])
    j = np.concatenate([f[1] for f in faces])
    flat = ids.ravel()
    rows = np.concatenate((flat, i, j))
    cols = np.concatenate((flat, j, i))

    def assemble(kappa, sigma):
        diag = vol * sigma.copy()
        weights = []
        for c, d, wall, h in faces:
            lc, ld = kappa[c], kappa[d]
            w = (vol / h**2) * (2.0 * lc * ld / (lc + ld))
            weights.append(w)
            diag[c] += w
            diag[d] += w
            kw = kappa[wall]
            # Add the two opposite walls separately if the axis has one cell.
            robin = (vol / h) * gamma * kw / (kw + 0.5 * h * gamma)
            if shape[len(weights) - 1] == 1:
                np.add.at(diag, wall, robin)
            else:
                diag[wall] += robin
        off = -np.concatenate(weights)
        return sparse.csr_matrix((np.concatenate((diag, off, off)), (rows, cols)),
                                 shape=(flat.size, flat.size))

    return vol, assemble


def _factor(a):
    return splu(a.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                options={'SymmetricMode': True, 'Equil': False})


def _tomography(s, z, y, lam_z, lam_s):
    # Represent P as a Gram operator without subtracting nearly equal energies.
    u, values, _ = linalg.svd(z, full_matrices=False)
    data = np.column_stack((s, y))
    coords = u.T @ data
    perpendicular = data - u @ coords
    shrink = np.sqrt(lam_z) / np.hypot(values, np.sqrt(lam_z))
    weighted = shrink[:, None] * coords
    m = s.shape[1]
    gram = perpendicular[:, :m].T @ perpendicular[:, :m] + weighted[:, :m].T @ weighted[:, :m]
    cross = perpendicular[:, :m].T @ perpendicular[:, m:] + weighted[:, :m].T @ weighted[:, m:]
    energy = np.diag(gram)
    profiled = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                         where=energy[:, None] > 0.0)
    gram.flat[::m + 1] += lam_s
    return profiled, linalg.solve(gram, cross, assume_a='pos')


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve all continuation states and fit snapshot amplitudes at the sensors."""
    base_grid = np.asarray(study['kappa_base'], dtype=float)
    shape = base_grid.shape
    spacing = np.array([study['Lx'], study['Ly'], study['Lz']]) / shape
    vol, assemble = _operators(shape, spacing, float(study['gamma_robin']))
    base = base_grid.ravel()
    mod1, mod2, sigma, smod = (np.asarray(study[key], dtype=float).ravel()
                               for key in ('kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    def operator(theta):
        return assemble(base + theta * mod1 + theta**2 * mod2, sigma + theta * smod)

    theta = np.asarray(study['thetas'], dtype=float)
    source = np.asarray(study['source_modes'], dtype=float)
    rhs = vol * source.reshape(source.shape[0], -1).T.copy(order='F')
    norms = np.linalg.norm(rhs, axis=0)
    sensors = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, shape)
    predictions = np.empty((theta.size, sensors.size, rhs.shape[1]))
    # A geometric center balances endpoint ratios for the quadratic coefficient.
    pivot = int(np.argmin(np.abs(np.log(theta) - np.log(theta).mean())))
    order = np.argsort(np.abs(np.log(theta / theta[pivot])))
    with threadpool_limits(limits=1, user_api='blas'):
        a_ref = operator(theta[pivot])
        factor = _factor(a_ref)
        factors = {pivot: factor}
        log_theta = np.log(theta)
        ref_fields = factor.solve(rhs)
        for k in order:
            a = a_ref if k == pivot else operator(theta[k])
            fields = ref_fields.copy()
            nearest = min(factors, key=lambda j: abs(log_theta[k] - log_theta[j]))
            preconditioner = LinearOperator(a.shape, matvec=factors[nearest].solve, dtype=float)
            local_factor = None
            for m, norm in enumerate(norms):
                if norm == 0:
                    continue
                b = rhs[:, m]
                u = fields[:, m]
                if k != pivot:
                    if local_factor is None:
                        u, _ = cg(a, b, x0=u, M=preconditioner, rtol=2e-11, atol=0., maxiter=45)
                    else:
                        u = local_factor.solve(b)
                residual = b - a @ u
                if np.linalg.norm(residual) > 2e-9 * norm:
                    if local_factor is None:
                        local_factor = factor if k == pivot else _factor(a)
                        factors[k] = local_factor
                    u = u + local_factor.solve(residual)
                fields[:, m] = u
            predictions[k] = fields[sensors]
            if k == theta.size - 1:
                terminal = fields.T.reshape(source.shape)
        stacked = predictions.reshape(-1, rhs.shape[1])
        profiled, joint = _tomography(stacked, np.asarray(study['Z_nuisance'], dtype=float),
                                      np.asarray(study['Y_obs'], dtype=float),
                                      float(study['nuisance_ridge']), float(study['source_ridge']))
    return {'terminal_fields': terminal, 'continuation_sensor_predictions': stacked,
            'profiled_amplitudes': profiled, 'joint_source_weights': joint}
