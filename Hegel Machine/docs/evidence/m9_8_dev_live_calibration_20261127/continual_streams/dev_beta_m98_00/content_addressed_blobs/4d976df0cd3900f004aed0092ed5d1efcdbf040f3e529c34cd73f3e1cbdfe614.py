"""Periodic-wall tomography with empirical nonlinear rosette calibration."""

import numpy as np
from scipy import sparse
from scipy.optimize import root
from scipy.sparse.linalg import splu


def _axis(k, v, h):
    """Periodic row-local upwind stencil with arithmetic face diffusivities."""
    n = len(k)
    i = np.arange(n)
    prev, nxt = (i - 1) % n, (i + 1) % n
    left = (k[i] + k[prev]) / (2 * h**2) + np.maximum(v, 0) / h
    right = (k[i] + k[nxt]) / (2 * h**2) + np.maximum(-v, 0) / h
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, prev, nxt])), shape=(n, n))


def _design(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ij = study['sensor_ij']
    rows = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * eye).solve(rhs)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.reshape(nx, ny, -1, order='F').transpose(2, 0, 1)


def _rosette_terms(x):
    return np.column_stack((np.ones(len(x)), x, x**3, x * np.roll(x, -1, axis=1)))


def _fit_rosette(ref, raw):
    # Gross offsets affect whole sweeps; screen them before estimating the
    # small cyclic interactions. Refit after residual screening on clean rows.
    terms = _rosette_terms(ref)
    amplitude = np.max(np.abs(raw), axis=1)
    keep = amplitude <= 4 * np.median(amplitude)
    for _ in range(4):
        fit = np.zeros((terms.shape[1], 4))
        for j in range(4):
            cols = [0, 1+j, 1+(j-1)%4, 1+(j+1)%4, 5+j, 9+j]
            fit[cols, j] = np.linalg.lstsq(terms[keep][:, cols], raw[keep, j], rcond=None)[0]
        residual = np.max(np.abs(terms @ fit - raw), axis=1)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        clean = residual <= center + 10 * max(mad, center, np.finfo(float).eps)
        if np.array_equal(clean, keep):
            break
        keep = clean
    return fit


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    fit = _fit_rosette(sweeps['ref_rosette'], sweeps['raw_rosette'])
    obs = study['y_obs_raw'].reshape(-1, 4)
    initial = np.linalg.solve(fit[1:5].T, (obs - fit[0]).T).T
    calibrated = []
    for y, x0 in zip(obs, initial):
        result = root(lambda x: (_rosette_terms(x[None, :]) @ fit)[0] - y,
                      x0, tol=1e-11)
        if np.max(np.abs(result.fun)) > 1e-9:
            raise RuntimeError('rosette inversion did not converge')
        calibrated.append(result.x)
    return np.asarray(calibrated).ravel()


def _box_qp(hessian, linear, cap, budget):
    """Primal active-set method; every blocking step preserves the sum."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    status = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(20 * n**2):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status)
        if len(free):
            nf = len(free)
            kkt = np.zeros((nf+1, nf+1))
            kkt[:nf, :nf] = hessian[np.ix_(free, free)]
            kkt[:nf, nf] = -1
            kkt[nf, :nf] = 1
            rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed],
                        budget - w[fixed].sum()]
            answer = np.linalg.solve(kkt, rhs)
            target, nu = answer[:-1], float(answer[-1])
            violation = np.flatnonzero((target < -1e-13) | (target > cap+1e-13))
            if len(violation):
                delta = target - w[free]
                bounds = np.where(target[violation] < 0, 0., cap)
                alpha = (bounds - w[free[violation]]) / delta[violation]
                i = int(np.argmin(alpha))
                w[free] += max(0., alpha[i]) * delta
                j = free[violation[i]]
                w[j] = bounds[i]
                status[j] = -1 if bounds[i] == 0 else 1
                continue
            w[free] = np.clip(target, 0., cap)
        else:
            g = hessian @ w + linear
            nu = float(g[status == 1].max())
        g = hessian @ w + linear
        violation = status * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= 1e-12:
            return w, nu
        status[j] = 0
    raise RuntimeError('Boxed source fit did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _design(study)
    y = _calibrate(study)
    z = study['Z_nuisance']
    projection = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    design -= z @ (projection @ design)
    y -= z @ (projection @ y)
    cap, budget, ridge = study['u_box'], study['total_budget'], study['source_ridge']
    w, nu = _box_qp(design.T @ design + ridge * np.eye(design.shape[1]), -design.T @ y, cap, budget)
    residual = design @ w - y
    g = design.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = float(objective - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., g - nu).tolist(),
                                'lam_upper': np.maximum(0., nu - g).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': lower}}
