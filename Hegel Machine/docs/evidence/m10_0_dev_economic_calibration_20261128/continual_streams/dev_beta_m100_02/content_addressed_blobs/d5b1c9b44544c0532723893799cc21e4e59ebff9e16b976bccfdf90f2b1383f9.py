"""Periodic-x, sealed-y transport inversion with robust, sweep-fitted rosette calibration."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    n = len(kappa)
    face = (np.asarray(kappa[:-1]) + kappa[1:]) / (2 * h**2)
    v = np.asarray(velocity)[1:-1] / h
    lower = np.empty(n - 1)
    upper = np.empty(n - 1)
    lower[:-1] = -face[:-1] - np.maximum(v, 0)
    upper[1:] = -face[1:] - np.maximum(-v, 0)
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    main = np.empty(n)
    main[1:-1] = -lower[:-1] - upper[1:]
    main[0] = -upper[0] + robin / h
    main[-1] = -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _periodic_axis(kappa, velocity, h):
    """Row-local finite volumes with a wrapped neighbor at each endpoint."""
    kappa = np.asarray(kappa)
    velocity = np.asarray(velocity)
    rows = np.arange(len(kappa))
    prev = (rows - 1) % len(rows)
    nxt = (rows + 1) % len(rows)
    left = (kappa + kappa[prev]) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + kappa[nxt]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    return csc_matrix((np.r_[left + right, -left, -right],
                       (np.tile(rows, 3), np.r_[rows, prev, nxt])),
                      shape=(len(rows), len(rows)))


def _operator(study):
    nx, ny = int(study['nx']), int(study['ny'])
    # Tracer pulses identify wraparound in x and no wall leakage in y.
    ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), 0.0)
    return (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
            + diags(np.ravel(study['V_reaction_2d'], order='F'), format='csc'))


def _responses(study, operator):
    nx, ny = int(study['nx']), int(study['ny'])
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    ij = np.asarray(study['sensor_ij'])
    rows = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    sampled = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[rows])
    return np.concatenate(sampled), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _sweep_fit(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    single = sweeps['single_bank_sweeps']
    coefficients = []
    for k in range(4):
        use = np.asarray(single['bank_id']).ravel() == k
        signal = np.asarray(single['ref_signal']).ravel()[use]
        isolated = np.zeros((len(signal), 4))
        isolated[:, k] = signal
        target = np.r_[raw[:, k], np.asarray(single['raw_readout']).ravel()[use]]
        x = np.vstack((ref, isolated))
        design = np.column_stack((np.ones(len(x)), x, x[:, k]**3, x[:, k] * x[:, (k + 1) % 4]))
        t0 = np.linalg.lstsq(design, target, rcond=None)[0]
        t = least_squares(lambda t: design @ t - target, t0, jac=lambda t: design.copy(),
                          loss='soft_l1', f_scale=.001).x
        e = abs(design @ t - target)
        keep = e < max(.001, 8 * np.median(e))
        t = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        coefficients.append(t)
    return np.asarray(coefficients)


def _raw(x, coefficients):
    return (coefficients[:, 0] + x @ coefficients[:, 1:5].T + coefficients[:, 5] * x**3
            + coefficients[:, 6] * x * np.roll(x, -1, axis=-1))


def _calibrate(study):
    t = _sweep_fit(study)
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    ids = np.arange(4)
    result = []
    for y in observed:
        def jac(x):
            j = t[:, 1:5].copy()
            j[ids, ids] += 3 * t[:, 5] * x**2 + t[:, 6] * np.roll(x, -1)
            j[ids, (ids + 1) % 4] += t[:, 6] * x
            return j
        x = np.linalg.solve(t[:, 1:5], y - t[:, 0])
        result.append(least_squares(lambda x: _raw(x, t) - y, x, jac=jac,
                                   gtol=1e-13, xtol=1e-13, ftol=1e-13).x)
    return np.asarray(result).ravel()


def _box_simplex(H, c, upper, total):
    """Feasible active set for a strictly convex box-simplex quadratic."""
    n = len(c)
    if total == 0:
        return np.zeros(n), float(np.min(c))
    if total == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w + c))
    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(np.linalg.norm(H, np.inf) * upper, np.linalg.norm(c, np.inf), 1.)
    for _ in range(max(100, 20 * n * n)):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if len(free):
            rhs = -c[free] - H[np.ix_(free, fixed)] @ w[fixed]
            reduced = H[np.ix_(free, free)]
            x, v = np.linalg.solve(reduced, np.column_stack((rhs, np.ones(len(free))))).T
            nu = (total - w[fixed].sum() - x.sum()) / v.sum()
            candidate = x + nu * v
            delta = candidate - w[free]
            crosses = np.where((candidate < 0) & (delta < 0), -1,
                               np.where((candidate > upper) & (delta > 0), 1, 0))
            bad = np.flatnonzero(crosses)
            if len(bad):
                ratios = (np.where(crosses[bad] > 0, upper, 0.) - w[free[bad]]) / delta[bad]
                pos = np.argmin(ratios)
                j = bad[pos]
                w[free] = np.clip(w[free] + ratios[pos] * delta, 0., upper)
                active[free[j]] = crosses[j]
                w[free[j]] = upper if crosses[j] > 0 else 0.
                continue
            w[free] = candidate
        g = H @ w + c
        if not len(free):
            lo = np.max(g[active > 0], initial=-np.inf)
            hi = np.min(g[active < 0], initial=np.inf)
            nu = (lo + hi) / 2
        violation = active * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _responses(study, _operator(study))
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((S, y))
    data -= Z @ np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = study['source_ridge']
    upper, total = study['u_box'], study['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, upper, total)
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    diff = g - nu
    dual = float(objective - g @ w + nu * total + upper * np.minimum(0., diff).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., diff).tolist(),
                                'lam_upper': np.maximum(0., -diff).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
