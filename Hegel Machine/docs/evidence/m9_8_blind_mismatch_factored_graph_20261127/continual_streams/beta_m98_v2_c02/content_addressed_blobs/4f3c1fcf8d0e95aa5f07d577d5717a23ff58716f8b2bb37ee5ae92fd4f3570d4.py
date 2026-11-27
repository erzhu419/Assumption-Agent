"""Steady transport tomography with periodic walls and empirical rosette calibration."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    """Periodic diffusion with shared face averages and row-local upwinding."""
    kappa = np.asarray(kappa)
    velocity = np.asarray(velocity)
    n = len(kappa)
    i = np.arange(n)
    face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
    left = np.roll(face, 1) + np.maximum(velocity, 0) / h
    right = face + np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, (i-1) % n, (i+1) % n])),
                             shape=(n, n))


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(reaction, format='csc'))
    ident = sparse.eye(nx * ny, format='csc')
    sources = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    ii, jj = np.asarray(study['sensor_ij']).T
    rows = ii + nx * jj
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(sources)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.T.reshape(-1, nx, ny, order='F')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    banks = sweeps['single_bank_sweeps']
    bank_ids = np.asarray(banks['bank_id'])
    bank_ref = np.asarray(banks['ref_signal'])
    bank_raw = np.asarray(banks['raw_readout'])
    calibrated = np.asarray(study['y_obs_raw'], dtype=float).reshape(-1, 4).copy()
    for channel in range(4):
        mask = bank_ids == channel
        x = np.r_[ref[:, channel], bank_ref[mask]]
        y = np.r_[raw[:, channel], bank_raw[mask]]
        centered = x - x.mean()
        gain = centered @ (y - y.mean()) / (centered @ centered)
        offset = y.mean() - gain * x.mean()
        calibrated[:, channel] = (calibrated[:, channel] - offset) / gain
    return calibrated.ravel()


def _box_budget_qp(H, b, cap, total):
    """Feasible active-set minimization of 0.5*w.T@H@w - b.T@w."""
    n = len(b)
    if total == 0 or total == n * cap:
        w = np.full(n, 0 if total == 0 else cap, dtype=float)
        g = H @ w - b
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)
    tol = 1e-12
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        g = H @ w - b
        if free.size:
            ones = np.ones(free.size)
            inv_g, inv_one = np.linalg.solve(H[np.ix_(free, free)], np.vstack((g[free], ones)).T).T
            nu = float(inv_g.sum() / inv_one.sum())
            p = nu * inv_one - inv_g
            to_bound = np.full(free.size, np.inf)
            down, up = p < -tol, p > tol
            to_bound[down] = -w[free[down]] / p[down]
            to_bound[up] = (cap - w[free[up]]) / p[up]
            j = np.argmin(to_bound)
            if to_bound[j] < 1:
                w[free] += max(0., to_bound[j]) * p
                i = free[j]
                active[i] = 1 if p[j] > 0 else -1
                w[i] = cap if active[i] == 1 else 0.
                continue
            w[free] += p
            g = H @ w - b
        else:
            nu = float((g[active == 1].max() + g[active == -1].min()) / 2)
        # A lower bound needs g >= nu; an upper bound needs g <= nu.
        violation = -active * (g - nu)
        i = np.argmin(violation)
        if violation[i] >= -tol:
            return w, nu
        active[i] = 0
    raise RuntimeError('Box-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    S, last = _forward(study)
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((S, y))
    P = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    projected = data - Z @ (P @ data)
    A, target = projected[:, :-1], projected[:, -1]
    ridge, cap, total = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = A.T @ A + ridge * np.eye(S.shape[1])
    w, nu = _box_budget_qp(H, A.T @ target, cap, total)
    residual = A @ w - target
    value = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = A.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    bound = float(value - g @ w + nu * total - cap * upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': value, 'dual_lower_bound': bound}}
