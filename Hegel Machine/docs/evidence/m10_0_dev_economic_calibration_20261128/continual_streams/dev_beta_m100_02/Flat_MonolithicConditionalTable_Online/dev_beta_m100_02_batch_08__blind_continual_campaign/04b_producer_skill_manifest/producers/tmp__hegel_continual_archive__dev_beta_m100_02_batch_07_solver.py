"""Transport inversion for the tracer-identified doubly periodic chamber."""

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _periodic_axis(kappa, velocity, h):
    rows = np.arange(len(kappa))
    prev = (rows - 1) % len(rows)
    nxt = (rows + 1) % len(rows)
    left = (kappa + kappa[prev]) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + kappa[nxt]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    return csc_matrix((np.r_[left + right, -left, -right],
                       (np.tile(rows, 3), np.r_[rows, prev, nxt])),
                      shape=(len(rows), len(rows)))


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _periodic_axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    modes = study['source_modes']
    rhs = modes.reshape((len(modes), nx * ny), order='F').T
    ii, jj = study['sensor_ij'].T
    rows = ii + nx * jj
    identity = eye(nx * ny, format='csc')
    sampled = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[rows])
    return np.concatenate(sampled), field.T.reshape(modes.shape, order='F')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    single = sweeps['single_bank_sweeps']
    decoded = np.empty_like(study['y_obs_raw'].reshape(-1, 4), dtype=float)
    for j in range(4):
        use = single['bank_id'] == j
        x = np.r_[ref[:, j], single['ref_signal'][use]]
        y = np.r_[raw[:, j], single['raw_readout'][use]]
        gain, offset = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y,
                                      rcond=None)[0]
        decoded[:, j] = (study['y_obs_raw'][j::4] - offset) / gain
    return decoded.ravel()


def _box_simplex(H, c, upper, total):
    """Minimize w.T @ H @ w / 2 + c @ w with a feasible active set."""
    n = len(c)
    if total == 0 or total == n * upper:
        w = np.full(n, 0.0 if total == 0 else upper)
        g = H @ w + c
        return w, float(g.min() if total == 0 else g.max())

    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)  # -1 lower, +1 upper
    for _ in range(100 * n):
        f = np.flatnonzero(active == 0)
        g = H @ w + c
        if not f.size:
            il = np.where(active < 0, g, np.inf).argmin()
            iu = np.where(active > 0, g, -np.inf).argmax()
            if g[iu] <= g[il]:
                return w, float((g[iu] + g[il]) / 2)
            active[[il, iu]] = 0
            continue

        # Solve for the exact equality-constrained minimizer on this face.
        fixed = np.flatnonzero(active)
        rhs = -c[f] - H[np.ix_(f, fixed)] @ w[fixed]
        a, b = np.linalg.solve(H[np.ix_(f, f)], np.column_stack((rhs, np.ones(len(f))))).T
        nu = (total - w[fixed].sum() - a.sum()) / b.sum()
        target = a + nu * b
        limits = np.full(len(f), np.inf)
        lo, hi = target < -1e-13, target > upper + 1e-13
        limits[lo] = -w[f[lo]] / (target[lo] - w[f[lo]])
        limits[hi] = (upper - w[f[hi]]) / (target[hi] - w[f[hi]])
        k = limits.argmin()
        if np.isfinite(limits[k]):
            w[f] += limits[k] * (target - w[f])
            active[f[k]] = 1 if hi[k] else -1
            w[f[k]] = upper if hi[k] else 0.0
            continue

        w[f] = np.clip(target, 0.0, upper)
        g = H @ w + c
        violation = active * (g - nu)
        j = violation.argmax()
        if violation[j] <= 1e-12:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _responses(study)
    y = _calibrate(study)
    Z = study['Z_nuisance']
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    St = S - Z @ (Pz @ S)
    ry = y - Z @ (Pz @ y)
    ridge, upper, total = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    H = St.T @ St + ridge * np.eye(St.shape[1])
    w, nu = _box_simplex(H, -St.T @ ry, upper, total)
    residual = St @ w - ry
    g = St.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    d = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {
                'weights': w.tolist(), 'nu_sum': nu,
                'lam_lower': np.maximum(0, d).tolist(),
                'lam_upper': np.maximum(0, -d).tolist(),
                'primal_objective': objective,
                'dual_lower_bound': float(objective - g @ w + nu * total
                                          + upper * np.minimum(0, d).sum()),
            }}
