"""Transport tomography with empirical rosette calibration and a boxed source fit."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(k, v, h):
    """Upwind interior with zero leakage at both walls."""
    n = len(k)
    face = (k[1:] + k[:-1]) / (2 * h**2)
    left = np.r_[-face[:-1] - np.maximum(v[1:-1], 0) / h, -k[-1] / h**2]
    right = np.r_[-k[0] / h**2, -face[1:] - np.maximum(-v[1:-1], 0) / h]
    center = np.r_[-right[0], -left[:-1] - right[1:], -left[-1]]
    return sparse.diags((left, center, right), (-1, 0, 1), shape=(n, n), format='csc')


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


def _rosette_terms(ref, channel):
    own = ref[:, channel]
    following = ref[:, (channel + 1) % 4]
    preceding = ref[:, (channel - 1) % 4]
    return np.column_stack((np.ones(len(ref)), own, following, preceding,
                            own * following, own**3))


def _fit_rosette(ref, raw):
    # Identify whole-sweep corruption before estimating nonlinear terms.
    seed = np.column_stack((np.ones(len(ref)), ref))
    residual = np.linalg.norm(raw - seed @ np.linalg.lstsq(seed, raw, rcond=None)[0], axis=1)
    clean = residual <= max(6 * np.median(residual), 1e-12)
    return np.column_stack([
        np.linalg.lstsq(_rosette_terms(ref[clean], i), raw[clean, i], rcond=None)[0]
        for i in range(4)
    ])


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bias, gain, forward, backward, mixed, cubic = _fit_rosette(
        sweeps['ref_rosette'], sweeps['raw_rosette'])
    raw = study['y_obs_raw'].reshape(-1, 4)
    i = np.arange(4)
    nxt, prev = (i + 1) % 4, (i - 1) % 4
    jac = np.zeros((len(raw), 4, 4))
    ref = np.zeros_like(raw, dtype=float)
    for _ in range(12):
        after = ref[:, nxt]
        residual = (bias + gain * ref + forward * after + backward * ref[:, prev]
                    + mixed * ref * after + cubic * ref**3 - raw)
        jac[:, i, i] = gain + mixed * after + 3 * cubic * ref**2
        jac[:, i, nxt] = forward + mixed * ref
        jac[:, i, prev] = backward
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        ref -= delta
        if np.max(np.abs(delta)) < 1e-12:
            break
    return ref.ravel()


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
