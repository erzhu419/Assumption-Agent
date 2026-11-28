"""Steady transport inversion for this campaign’s sealed chamber."""

import numpy as np
from scipy.sparse import diags, eye, kron
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


def _operator(study):
    nx, ny = int(study['nx']), int(study['ny'])
    # The public tracer pulses identify zero wall leakage on both axes.
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), 0.0)
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


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    design = np.column_stack((ref, np.ones(len(ref))))
    single = sweeps['single_bank_sweeps']
    bank = np.asarray(single['bank_id']).ravel()
    signal = np.asarray(single['ref_signal']).ravel()
    readout = np.asarray(single['raw_readout']).ravel()
    coeff = np.empty((5, 4))
    for k in range(4):
        use = bank == k
        # Only the driven bank has a nonzero reference in these sweeps.
        isolated = np.zeros((np.count_nonzero(use), 5))
        isolated[:, k] = signal[use]
        isolated[:, -1] = 1.0
        coeff[:, k] = np.linalg.lstsq(
            np.vstack((design, isolated)), np.r_[raw[:, k], readout[use]], rcond=None
        )[0]
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coeff[:4].T, (observed - coeff[4]).T).T.ravel()


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
