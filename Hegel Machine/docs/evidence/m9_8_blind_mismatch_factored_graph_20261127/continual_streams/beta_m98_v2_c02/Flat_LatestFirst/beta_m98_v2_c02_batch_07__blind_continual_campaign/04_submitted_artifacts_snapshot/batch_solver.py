"""Periodic steady transport tomography with sweep-fitted rosette calibration."""

import numpy as np
from scipy import sparse
from scipy.optimize import root
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    kappa = np.asarray(kappa)
    velocity = np.asarray(velocity)
    i = np.arange(kappa.size)
    left, right = (i - 1) % i.size, (i + 1) % i.size
    km = 0.5 * (kappa + kappa[left]) / h**2
    kp = 0.5 * (kappa + kappa[right]) / h**2
    lower = -km - np.maximum(velocity, 0) / h
    upper = -kp - np.maximum(-velocity, 0) / h
    return sparse.coo_matrix((np.r_[lower, -lower - upper, upper],
                             (np.tile(i, 3), np.r_[left, i, right])),
                            shape=(i.size, i.size)).tocsc()


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


def _rosette_basis(ref, bank):
    own = ref[..., bank]
    prev = ref[..., (bank - 1) % 4]
    nxt = ref[..., (bank + 1) % 4]
    return np.stack((np.ones_like(own), own, prev, nxt, own * nxt, own**3), axis=-1)


def _fit_rosette(sweeps):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    design = [_rosette_basis(ref, j) for j in range(4)]
    # Gross corruption affects all four readouts of a paired sweep.
    valid = np.ones(len(ref), dtype=bool)
    while True:
        coef = np.stack([np.linalg.lstsq(A[valid], raw[valid, j], rcond=None)[0]
                         for j, A in enumerate(design)])
        error = np.max(np.abs(np.column_stack([A @ c for A, c in zip(design, coef)]) - raw), axis=1)
        tol = max(5 * np.median(error[valid]), 1e-9)
        keep = valid & (error <= tol)
        if np.array_equal(keep, valid):
            break
        valid = keep

    single = sweeps['single_bank_sweeps']
    banks = np.asarray(single['bank_id'])
    solo_ref = np.zeros((len(banks), 4))
    solo_ref[np.arange(len(banks)), banks] = single['ref_signal']
    solo_raw = np.asarray(single['raw_readout'])
    for j, A in enumerate(design):
        B = _rosette_basis(solo_ref[banks == j], j)
        z = solo_raw[banks == j]
        good = np.abs(B @ coef[j] - z) <= tol
        coef[j] = np.linalg.lstsq(np.vstack((A[valid], B[good])),
                                 np.r_[raw[valid, j], z[good]], rcond=None)[0]
    return coef


def _rosette_raw(ref, coef):
    return np.stack([_rosette_basis(ref, j) @ coef[j] for j in range(4)], axis=-1)


def _calibrate(study):
    coef = _fit_rosette(study['sensor_calibration_sweeps'])
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    linear = (_rosette_raw(np.eye(4), coef * [1, 1, 1, 1, 0, 0]) - coef[:, 0]).T
    initial = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
    calibrated = np.empty_like(raw)
    for i, (row, guess) in enumerate(zip(raw, initial)):
        def residual(ref):
            return _rosette_raw(ref, coef) - row
        sol = root(residual, guess, tol=1e-11)
        if np.max(np.abs(residual(sol.x))) > 1e-9:
            raise RuntimeError('Sensor rosette inversion did not converge')
        calibrated[i] = sol.x
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
