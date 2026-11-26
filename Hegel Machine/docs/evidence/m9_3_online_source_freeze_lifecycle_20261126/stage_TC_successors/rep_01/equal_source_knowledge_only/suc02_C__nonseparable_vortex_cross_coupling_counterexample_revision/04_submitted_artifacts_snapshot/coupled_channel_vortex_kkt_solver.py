"""Forward transport and certified source inversion for the two-channel survey."""
import numpy as np
from scipy.linalg import eigh_tridiagonal, helmert
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(length, diffusivity, velocity, robin):
    """Cell-centered harmonic diffusion, upwind advection, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    center = np.abs(v) / h
    center[:-1] += face
    center[1:] += face
    for i in (0, k.size - 1):
        center[i] += robin * k[i] / (h * (k[i] + 0.5 * robin * h))
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face - np.maximum(-v[:-1], 0) / h
    return center, lower, upper


def _spectrum(axis):
    # D^-1 A D is symmetric; center the log of D without changing the similarity.
    center, lower, upper = axis
    log_scale = np.r_[0., np.cumsum(0.5 * (np.log(-lower) - np.log(-upper)))]
    scale = np.exp(log_scale - np.mean(log_scale))
    eigenvalues, basis = eigh_tridiagonal(center, -np.sqrt((-lower) * (-upper)))
    return eigenvalues, basis, scale


def _readouts(study):
    ax, by = [_axis(study['L' + d], study['kappa_' + d], study['vel_' + d],
                    study['robin_' + d]) for d in ('x', 'y')]
    ex, qx, dx = _spectrum(ax)
    ey, qy, dy = _spectrum(by)
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    scale = dx[:, None] * dy[None, :]
    transformed = qx.T @ (modes / scale) @ qy
    eigen_sums = ex[:, None] + ey[None, :]

    ax_sp, by_sp = [diags(axis, (0, -1, 1), shape=(len(axis[0]), len(axis[0])),
                          format='csc') for axis in (ax, by)]
    eye_x, eye_y = eye(nx, format='csc'), eye(ny, format='csc')
    potential = np.asarray(study['V_pocket_channel_b'], dtype=float).ravel()
    coupled = (kron(ax_sp, eye_y, format='csc') + kron(eye_x, by_sp, format='csc')
               + diags(potential, format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = modes.reshape(m, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    rows_a, rows_b = [], []
    for shift in study['mu_shifts']:
        fields_a = (qx @ (transformed / (eigen_sums + shift)) @ qy.T) * scale
        fields_b = splu(csc_matrix(coupled + shift * identity)).solve(rhs).T.reshape(m, nx, ny)
        rows_a.append(fields_a[:, ix, iy].T)
        rows_b.append(fields_b[:, ix, iy].T)
    return np.vstack(rows_a + rows_b), fields_a, fields_b


def _profile(design, observations, nuisance, ridge):
    """Apply the ridge residual maker to both the sensor matrix and observations."""
    values = np.column_stack((design, observations))
    if ridge > 0:
        gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
        values -= nuisance @ np.linalg.solve(gram, nuisance.T @ values)
    elif nuisance.shape[1]:
        values -= nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]
    return values[:, :-1], values[:, -1]


def _certificate(design, target, ridge, w, cap, budget):
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    # This order statistic maximizes the affine minorant over equality multipliers.
    k = min(w.size - 1, max(0, int(np.ceil(budget / cap)) - 1)) if cap else 0
    nu = float(np.partition(g, k)[k])
    lower = np.maximum(g - nu, 0.)
    upper = np.maximum(nu - g, 0.)
    primal = 0.5 * float(residual @ residual + ridge * (w @ w))
    gap = float(lower @ w + upper @ (cap - w) + nu * (w.sum() - budget))
    return {'weights': w, 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
            'primal_objective': primal,
            'dual_lower_bound': primal - max(0., gap)}, gap


def _solve_weights(design, target, ridge, cap, budget):
    """Minimize on a box face, then add blocking bounds or release violated ones."""
    m = design.shape[1]
    if budget == 0 or budget == m * cap:
        w = np.full(m, 0. if budget == 0 else cap)
        return w, _certificate(design, target, ridge, w, cap, budget)[0]
    a = np.vstack((design, np.sqrt(ridge) * np.eye(m)))
    y = np.r_[target, np.zeros(m)]
    w = np.full(m, budget / m)
    bound = np.zeros(m, dtype=int)
    snap = 8 * np.finfo(float).eps * cap
    for _ in range(40 * m):
        free = np.flatnonzero(bound == 0)
        if not free.size:
            g = a.T @ (a @ w - y)
            lo, hi = np.flatnonzero(bound == -1), np.flatnonzero(bound == 1)
            bound[[lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]]] = 0
            continue
        trial = w.copy()
        trial[free] = (budget - cap * np.count_nonzero(bound == 1)) / free.size
        if free.size > 1:
            tangent = helmert(free.size).T
            trial[free] += tangent @ np.linalg.lstsq(a[:, free] @ tangent,
                                                    y - a @ trial, rcond=None)[0]
        hits = free[(trial[free] < -snap) | (trial[free] > cap + snap)]
        if hits.size:
            delta = trial - w
            limits = np.where(trial[hits] < 0., 0., cap)
            steps = (limits - w[hits]) / delta[hits]
            j = np.argmin(steps)
            i = hits[j]
            w = np.clip(w + np.clip(steps[j], 0., 1.) * delta, 0., cap)
            w[i] = limits[j]
            bound[i] = -1 if limits[j] == 0. else 1
        else:
            w = np.clip(trial, 0., cap)
            cert, gap = _certificate(design, target, ridge, w, cap, budget)
            if abs(w.sum() - budget) <= 1e-12 * max(1., budget) and abs(gap) <= 1e-10:
                return w, cert
            g = a.T @ (a @ w - y)
            violation = bound * (g - np.mean(g[free]))
            i = np.argmax(violation)
            if violation[i] > 0.:
                bound[i] = 0
    raise RuntimeError('Box-simplex solve did not attain the requested KKT gap')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Solve both channels and fit their common source mixture."""
    design, last_a, last_b = _readouts(study)
    design, target = _profile(design, np.asarray(study['y_obs'], dtype=float).ravel(),
                              np.asarray(study['Z_nuisance'], dtype=float),
                              float(study['nuisance_ridge']))
    w, cert = _solve_weights(design, target, float(study['source_ridge']),
                             float(study['u_box']), float(study['total_budget']))
    return {'weights': w,
            'terminal_field_channel_a': np.einsum('m,mxy->xy', w, last_a),
            'terminal_field_channel_b': np.einsum('m,mxy->xy', w, last_b),
            'kkt_certificate': cert}
