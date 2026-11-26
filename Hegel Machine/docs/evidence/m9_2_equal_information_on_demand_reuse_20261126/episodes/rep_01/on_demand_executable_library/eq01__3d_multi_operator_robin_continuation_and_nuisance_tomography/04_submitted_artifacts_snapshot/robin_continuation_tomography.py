"""Cell-centered Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext

    def threadpool_limits(**kwargs):
        return nullcontext()


class _RobinOperator:
    def __init__(self, shape, lengths, gamma):
        self.shape = shape
        self.h = np.asarray(lengths, dtype=float) / shape
        self.volume = float(np.prod(self.h))
        self.gamma = gamma
        size = int(np.prod(shape))
        ids = np.arange(size).reshape(shape)
        self.faces = []
        self.walls = []
        for axis, h in enumerate(self.h):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            self.faces.append((ids[tuple(lo)].ravel(), ids[tuple(hi)].ravel(),
                               self.volume / h**2))
            self.walls.append((np.take(ids, [0, shape[axis] - 1], axis=axis).ravel(), h))
        left = np.concatenate([f[0] for f in self.faces])
        right = np.concatenate([f[1] for f in self.faces])
        cells = ids.ravel()
        self.rows = np.concatenate((cells, left, right))
        self.cols = np.concatenate((cells, right, left))

    def assemble(self, kappa, sigma):
        kappa = kappa.ravel()
        diag = self.volume * sigma.ravel()
        off = []
        for i, j, scale in self.faces:
            w = scale * (2 * kappa[i] * kappa[j] / (kappa[i] + kappa[j]))
            diag[i] += w
            diag[j] += w
            off.append(-w)
        for ids, h in self.walls:
            k = kappa[ids]
            wall = (self.volume / h) * (self.gamma * k / (k + self.gamma * h / 2))
            np.add.at(diag, ids, wall)
        values = np.concatenate([diag] + off + off)
        return sparse.csr_matrix((values, (self.rows, self.cols)),
                                 shape=(diag.size, diag.size))


def _profile(sensors, observations, nuisance, ridge_z, ridge_s):
    # An orthogonal transform eliminates the nuisance columns from the
    # augmented least-squares problem without subtracting nearly equal Grams.
    n, p = nuisance.shape
    m = sensors.shape[1]
    rhs = np.zeros((n + p, m + observations.shape[1]), order='F')
    rhs[:n, :m] = sensors
    rhs[:n, m:] = observations
    if p:
        design = np.empty((n + p, p), order='F')
        design[:n] = nuisance
        design[n:] = np.sqrt(ridge_z) * np.eye(p)
        (reflectors, tau), _ = linalg.qr(design, mode='raw', overwrite_a=True,
                                         check_finite=False)
        ormqr = linalg.get_lapack_funcs('ormqr', (reflectors, rhs))
        _, work, _ = ormqr('L', 'T', reflectors, tau, rhs, -1)
        rhs, _, _ = ormqr('L', 'T', reflectors, tau, rhs, int(work[0]), overwrite_c=1)
    s, y = rhs[p:, :m], rhs[p:, m:]
    energy = np.einsum('ij,ij->j', s, s)
    cross = s.T @ y
    amplitudes = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                           where=energy[:, None] > 0)
    # Solve the small joint source problem directly in square-root form.
    design_s = np.vstack((s, np.sqrt(ridge_s) * np.eye(m)))
    target_s = np.vstack((y, np.zeros((m, y.shape[1]))))
    q, r = linalg.qr(design_s, mode='economic', check_finite=False)
    weights = linalg.solve_triangular(r, q.T @ target_s, check_finite=False)
    return amplitudes, weights


def _solve_state(a, rhs, rhs_norm, fields, inverse):
    direct = None
    for m, b in enumerate(rhs):
        norm = rhs_norm[m]
        if norm == 0:
            continue
        u, _ = cg(a, b, x0=fields[m], M=inverse, rtol=1e-10, maxiter=180)
        # Check the actual residual, allowing for cancellation in the nearly
        # constant solution component when reaction and wall transfer are weak.
        r = b - a @ u
        error = np.linalg.norm(r)
        if error > 7e-8 * norm:
            du, _ = cg(a, r, M=inverse, rtol=1e-3, maxiter=100)
            candidate = u + du
            improved = np.linalg.norm(b - a @ candidate)
            if improved < error:
                u, error = candidate, improved
        if error > 7e-8 * norm:
            if direct is None:
                direct = splu(a.tocsc(), permc_spec='MMD_AT_PLUS_A',
                              diag_pivot_thresh=0, options={'SymmetricMode': True})
            candidate = direct.solve(b)
            for _ in range(3):
                r = b - a @ candidate
                improved = np.linalg.norm(r)
                if improved < error:
                    u, error = candidate, improved
                if error <= 7e-8 * norm:
                    break
                candidate = candidate + direct.solve(r)
        fields[m] = u
    return fields


def _solve(study):
    kb, k1, k2, sb, s1 = (np.asarray(study[key], dtype=float) for key in
                         ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    grid = _RobinOperator(kb.shape, [study[key] for key in ('Lx', 'Ly', 'Lz')],
                          float(study['gamma_robin']))
    theta = np.asarray(study['thetas'], dtype=float)
    modes = np.asarray(study['source_modes'], dtype=float).reshape(-1, kb.size)
    rhs = grid.volume * modes
    rhs_norm = np.linalg.norm(rhs, axis=1)
    indices = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, kb.shape)
    predictions = np.empty((theta.size, indices.size, modes.shape[0]))

    def matrix(t):
        return grid.assemble(kb + t * k1 + t**2 * k2, sb + t * s1)

    # Balance the theta/theta_ref ratios for the positive diffusion terms.
    t_ref = np.sqrt(theta.min() * theta.max())
    reference = matrix(t_ref)
    factor = splu(reference.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                  options={'SymmetricMode': True})
    inverse = LinearOperator(reference.shape, matvec=factor.solve, dtype=float)
    fields = np.zeros_like(rhs)
    last = np.empty_like(fields)
    for k in np.argsort(theta):
        a = matrix(theta[k])
        fields = _solve_state(a, rhs, rhs_norm, fields, inverse)
        predictions[k] = fields[:, indices].T
        if k == theta.size - 1:
            last[:] = fields

    stacked = predictions.reshape(-1, modes.shape[0])
    amplitudes, weights = _profile(stacked, np.asarray(study['Y_obs'], dtype=float),
                                   np.asarray(study['Z_nuisance'], dtype=float),
                                   study['nuisance_ridge'], study['source_ridge'])
    return {'terminal_fields': last.reshape((-1,) + kb.shape),
            'continuation_sensor_predictions': stacked,
            'profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve the continuation family and fit independent and joint source weights."""
    with threadpool_limits(limits=1, user_api='blas'):
        return _solve(study)
