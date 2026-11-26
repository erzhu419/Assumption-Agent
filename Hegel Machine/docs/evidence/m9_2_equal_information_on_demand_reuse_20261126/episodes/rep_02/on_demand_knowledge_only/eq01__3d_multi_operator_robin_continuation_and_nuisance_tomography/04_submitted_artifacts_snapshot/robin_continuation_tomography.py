"""Cell-centered Robin diffusion continuation and ridge-profiled tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits


def _operator(study, theta, grid, spacing):
    """Assemble conservative interior fluxes and six half-cell wall fluxes."""
    volume = np.prod(spacing)
    kappa = study['kappa_base'] + theta * study['kappa_mod1'] + theta**2 * study['kappa_mod2']
    diagonal = volume * (study['sigma_base'] + theta * study['sigma_mod'])
    gamma = study['gamma_robin']
    rows, cols, values = [], [], []
    for axis, h in enumerate(spacing):
        area = volume / h
        low = np.swapaxes(kappa, 0, axis)
        ids = np.swapaxes(grid, 0, axis)
        diag = np.swapaxes(diagonal, 0, axis)
        left, right = low[:-1], low[1:]
        w = (area / h) * (2 * left * right / (left + right))
        diag[:-1] += w
        diag[1:] += w
        for wall in (0, -1):
            face = low[wall]
            diag[wall] += area * (gamma * face) / (gamma * 0.5 * h + face)
        i, j = ids[:-1].ravel(), ids[1:].ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-w.ravel(), -w.ravel()))
    ids = grid.ravel()
    rows.append(ids)
    cols.append(ids)
    values.append(diagonal.ravel())
    return sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(grid.size, grid.size))


def _profile(sensors, observations, nuisance, lam_z, lam_s):
    # Orthogonal complement and shrunken nuisance directions give a Gram
    # representation of the ridge residual, without subtracting two Grams.
    u, singular, _ = linalg.svd(nuisance, full_matrices=False, check_finite=False)
    data = np.column_stack((sensors, observations))
    coordinates = u.T @ data
    shrink = np.sqrt(lam_z / (singular**2 + lam_z))
    whitened = np.concatenate((data - u @ coordinates, shrink[:, None] * coordinates), axis=0)
    modes = sensors.shape[1]
    s, y = whitened[:, :modes], whitened[:, modes:]
    gram = s.T @ s
    cross = s.T @ y
    amplitudes = cross / gram.diagonal()[:, None]
    # Augmented least squares also handles nearly collinear source modes.
    a = np.concatenate((s, np.sqrt(lam_s) * np.eye(modes)), axis=0)
    b = np.concatenate((y, np.zeros((modes, y.shape[1]))), axis=0)
    weights = linalg.lstsq(a, b, check_finite=False)[0]
    return amplitudes, weights


def solve_robin_continuation_tomography(study: dict) -> dict:
    shape = study['kappa_base'].shape
    spacing = np.asarray([study['Lx'], study['Ly'], study['Lz']]) / shape
    grid = np.arange(np.prod(shape)).reshape(shape)
    thetas = np.asarray(study['thetas'])
    sources = np.asarray(study['source_modes'])
    rhs = np.ascontiguousarray(sources.reshape(len(sources), -1) * np.prod(spacing))
    sensor_ids = grid[tuple(np.asarray(study['sensor_ijk']).T)]
    stack = np.empty((len(thetas), len(sensor_ids), len(rhs)))
    fields = np.empty((len(rhs), grid.size))
    terminal = np.empty_like(fields)
    # The geometric midpoint controls multiplicative changes in positive
    # coefficient terms at both ends of the continuation interval.
    ref = float(np.sqrt(thetas.min() * thetas.max()))
    with threadpool_limits(limits=1, user_api='blas'):
        reference = _operator(study, ref, grid, spacing)
        factor = splu(reference.tocsc(), permc_spec='MMD_AT_PLUS_A',
                      diag_pivot_thresh=0, options={'SymmetricMode': True})
        preconditioner = LinearOperator(reference.shape, matvec=factor.solve, dtype=float)
        norms = np.linalg.norm(rhs, axis=1)
        # Keep neighboring warm starts close without changing the output order.
        for step, k in enumerate(np.argsort(thetas)):
            theta = thetas[k]
            operator = reference if theta == ref else _operator(study, float(theta), grid, spacing)
            for m, b in enumerate(rhs):
                if norms[m] == 0:
                    fields[m].fill(0)
                    continue
                x, _ = cg(operator, b, x0=fields[m] if step else None,
                          M=preconditioner, rtol=1e-10, maxiter=200)
                residual = b - operator @ x
                if np.linalg.norm(residual) > 5e-8 * norms[m]:
                    delta, _ = cg(operator, residual, M=preconditioner,
                                  rtol=1e-5, maxiter=400)
                    x += delta
                    if np.linalg.norm(b - operator @ x) > 1e-7 * norms[m]:
                        raise RuntimeError(f'Robin solve failed residual target at state {k}, mode {m}')
                fields[m] = x
            stack[k] = fields[:, sensor_ids].T
            if k == len(thetas) - 1:
                terminal[:] = fields
        sensors = stack.reshape(-1, len(rhs))
        amplitudes, weights = _profile(sensors, study['Y_obs'], study['Z_nuisance'],
                                       study['nuisance_ridge'], study['source_ridge'])
    return {'terminal_fields': terminal.reshape(sources.shape),
            'continuation_sensor_predictions': sensors,
            'profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}
