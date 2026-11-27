"""Periodic-x, sealed-y transport inversion on a box-constrained source simplex."""
from itertools import combinations

import numpy as np
from scipy.optimize import root
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    """Row-local upwind interior with zero-leak endpoint rows."""
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h, -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2, -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[kappa[0] / h**2, -lower[:-1] - upper[1:], kappa[-1] / h**2]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _periodic_axis(kappa, velocity, h):
    """Conservative row-local upwind stencil with periodic endpoint coupling."""
    n = len(kappa)
    i = np.arange(n)
    west, east = (i - 1) % n, (i + 1) % n
    lower = -(kappa + kappa[west]) / (2 * h**2) - np.maximum(velocity, 0) / h
    upper = -(kappa + kappa[east]) / (2 * h**2) - np.maximum(-velocity, 0) / h
    return csc_matrix((np.r_[lower, -lower-upper, upper],
                       (np.tile(i, 3), np.r_[west, i, east])), shape=(n, n))


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.reshape(len(modes), -1, order='F').T.copy()
    i, j = np.asarray(study['sensor_ij']).T
    sensor = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sensor])
    return np.concatenate(blocks), fields.T.reshape(len(modes), nx, ny, order='F')


_PAIRS = tuple(combinations(range(4), 2))


def _rosette_design(x):
    return np.column_stack((np.ones(len(x)), x, x**3,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS)))


def _fit_rosette(ref, raw, single_bank_sweeps=None):
    # Use robust pilot residuals to exclude corrupted sweeps as whole rosettes.
    design = _rosette_design(ref)
    coeff = np.linalg.lstsq(design, raw, rcond=None)[0]
    for _ in range(35):
        residual = raw - design @ coeff
        centered = residual - np.median(residual, axis=0)
        scale = np.maximum(1.4826 * np.median(abs(centered), axis=0), 1e-7)
        factor = 1 / np.hypot(1, centered / (2.5 * scale))
        for j in range(raw.shape[1]):
            f = factor[:, j]
            coeff[:, j] = np.linalg.lstsq(design * f[:, None], raw[:, j] * f, rcond=None)[0]
    residual = raw - design @ coeff
    centered = residual - np.median(residual, axis=0)
    scale = np.maximum(1.4826 * np.median(abs(centered), axis=0), 1e-7)
    clean = np.all(abs(centered) <= 8 * scale, axis=1)
    coeff = np.linalg.lstsq(design[clean], raw[clean], rcond=None)[0]
    if single_bank_sweeps is None:
        return coeff

    # Each bank sweep drives one input and measures that same output only.
    banks = np.asarray(single_bank_sweeps['bank_id'])
    bank_ref = np.zeros((len(banks), ref.shape[1]))
    bank_ref[np.arange(len(banks)), banks] = single_bank_sweeps['ref_signal']
    bank_design = _rosette_design(bank_ref)
    bank_raw = np.asarray(single_bank_sweeps['raw_readout'])
    residual = raw - design @ coeff
    centered = residual - np.median(residual, axis=0)
    scale = np.maximum(1.4826 * np.median(abs(centered), axis=0), 1e-7)
    clean = np.all(abs(centered) <= 8 * scale, axis=1)
    predicted = (bank_design @ coeff)[np.arange(len(banks)), banks]
    bank_clean = abs(bank_raw - predicted) <= 8 * scale[banks]
    for j in range(raw.shape[1]):
        selected = bank_clean & (banks == j)
        coeff[:, j] = np.linalg.lstsq(
            np.vstack((design[clean], bank_design[selected])),
            np.concatenate((raw[clean, j], bank_raw[selected])),
            rcond=None,
        )[0]
    return coeff


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coeff = _fit_rosette(sweeps['ref_rosette'], sweeps['raw_rosette'],
                         sweeps['single_bank_sweeps'])
    targets = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    linear = coeff[1:5].T
    start = np.linalg.solve(linear, (targets - coeff[0]).T).T

    def jacobian(x):
        jac = linear + 3 * coeff[5:9].T * x**2
        for k, (i, j) in enumerate(_PAIRS, 9):
            jac[:, i] += coeff[k] * x[j]
            jac[:, j] += coeff[k] * x[i]
        return jac

    recovered = np.empty_like(start)
    for i, (target, initial) in enumerate(zip(targets, start)):
        def residual(x):
            return _rosette_design(x[None])[0] @ coeff - target
        result = root(residual, initial, jac=jacobian, tol=1e-10)
        if np.max(abs(result.fun)) > 1e-9:
            raise RuntimeError('Rosette inversion did not converge')
        recovered[i] = result.x
    return recovered.ravel()


def _box_simplex(hessian, linear, cap, budget):
    """Primal active set, using exact equality-constrained solves on each face."""
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1: lower; +1: upper
    tol = 1e-12 * max(1., np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 20):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status)
        if not len(free):
            g = hessian @ w + linear
            lower, upper = np.flatnonzero(status == -1), np.flatnonzero(status == 1)
            a, b = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[a] >= g[b] - tol:
                return w, float((g[a] + g[b]) / 2)
            status[[a, b]] = 0
            continue
        hff = hessian[np.ix_(free, free)]
        cf = linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed]
        q, t = np.linalg.solve(hff, np.column_stack((-cf, np.ones(len(free))))).T
        nu = (budget - w[fixed].sum() - q.sum()) / t.sum()
        target = q + nu * t
        step = target - w[free]
        ratio = np.full(len(free), np.inf)
        lower, upper = (target < 0) & (step < 0), (target > cap) & (step > 0)
        ratio[lower] = -w[free[lower]] / step[lower]
        ratio[upper] = (cap - w[free[upper]]) / step[upper]
        k = int(ratio.argmin())
        if ratio[k] < 1:
            w[free] += ratio[k] * step
            status[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0. if lower[k] else cap
            continue
        w[free] = target
        g = hessian @ w + linear
        release = status * (g - nu)
        k = int(release.argmax())
        if release[k] <= tol:
            return w, float(nu)
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s, u_last = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    proj = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s -= z @ (proj @ s)
    y -= z @ (proj @ y)
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, u_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(reduced, 0).tolist(),
                                'lam_upper': np.maximum(-reduced, 0).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': float(dual)}}
