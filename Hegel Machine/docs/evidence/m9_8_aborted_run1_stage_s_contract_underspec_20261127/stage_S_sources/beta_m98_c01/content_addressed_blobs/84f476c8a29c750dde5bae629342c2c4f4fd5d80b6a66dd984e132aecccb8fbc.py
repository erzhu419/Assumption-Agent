"""Calibrated transport inversion and box-simplex optimality certificate."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    terms = {d: list(combinations_with_replacement(range(4), d)) for d in (2, 3)}

    def design(z, degree):
        cols = [np.ones(len(z)), *z.T]
        for d in range(2, degree + 1):
            cols.extend(z[:, t].prod(axis=1) for t in terms[d])
        return np.column_stack(cols)

    # Compare forward models on shared held-out sweeps; invert only the winner.
    degree, best = 1, np.inf
    fold = np.arange(len(x)) % 5
    for d in (1, 2, 3):
        a = design(x, d)
        if a.shape[1] >= 0.8 * len(x):
            continue
        mse = np.mean([np.mean((a[fold == f] @ np.linalg.lstsq(a[fold != f], y[fold != f], rcond=None)[0] - y[fold == f])**2)
                       for f in np.unique(fold)])
        if mse < 0.92 * best:
            degree, best = d, mse
    c = np.linalg.lstsq(design(x, degree), y, rcond=None)[0]

    def correct(raw):
        shape = np.shape(raw)
        raw = np.asarray(raw, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(c[1:5].T, (raw - c[0]).T).T
        if degree > 1:
            for i, row in enumerate(raw):
                sol = least_squares(lambda u: (design(u[None, :], degree) @ c)[0] - row,
                                    z[i], gtol=1e-12, ftol=1e-12, xtol=1e-12)
                z[i] = sol.x
        return z.reshape(shape)

    return correct


def _axis(k, v, h):
    d = (k[:-1] + k[1:]) * (0.5 / h**2)
    lo = d + np.maximum(v[1:], 0) / h
    hi = d + np.maximum(-v[:-1], 0) / h
    return diags((-lo, np.r_[0., lo] + np.r_[hi, 0.], -hi), (-1, 0, 1), format='csc')


def _chamber(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx'] / (nx - 1), s['Ly'] / (ny - 1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    a = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    wall = np.zeros((nx, ny, 4))
    wall[0, :, 0] = wall[-1, :, 1] = wall[:, 0, 2] = wall[:, -1, 3] = 1
    wall = wall.reshape(-1, 4, order='F')
    ij = np.asarray(s['sensor_ij'], dtype=int)
    at = ij[:, 0] + nx * ij[:, 1]
    ident = eye(nx * ny, format='csc')
    source = np.asarray(s['source_modes'], dtype=float).reshape(-1, nx * ny, order='F').T

    def response(walls, rhs=source, shift_offset=0.):
        operator = a + diags(wall @ walls)
        rows = []
        for shift in s['mu_shifts']:
            field = splu(operator + (shift + shift_offset) * ident).solve(rhs)
            rows.append(field[at])
        return np.concatenate(rows, axis=0), field.T.reshape(-1, nx, ny, order='F')

    return response, source, np.repeat([s['robin_x'] / hx, s['robin_y'] / hy], 2)


def _fit_walls(s, correct, response, source, nominal):
    pulses = s.get('reference_tracer_pulses')
    if not pulses:
        return nominal
    rhs = source @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([correct(p['y_pulse_raw']) for p in pulses])

    def residual(walls):
        return (response(walls, rhs)[0] - observed).ravel()

    return least_squares(residual, np.maximum(nominal, 1e-8), bounds=(0., np.inf),
                         gtol=1e-13, ftol=1e-11, xtol=1e-11).x


def _box_simplex(h, c, upper, budget):
    """Primal active-set QP, followed by a Lagrangian lower bound."""
    n = len(c)
    w = np.zeros(n) if not upper.sum() else upper * (budget / upper.sum())
    lock = upper == 0.
    state = np.zeros(n, dtype=int)
    if budget == 0:
        state[:] = -1
    elif budget == upper.sum():
        state[:] = 1
    for _ in range(max(100, 10*n*n)):
        free = (state == 0) & ~lock
        g = h @ w + c
        if np.any(free):
            hf = h[np.ix_(free, free)]
            invg, inv1 = np.linalg.solve(hf, np.column_stack((g[free], np.ones(free.sum())))).T
            nu = -invg.sum() / inv1.sum()
            p = -invg - nu * inv1
            ix = np.flatnonzero(free)
            ratios = np.full(len(ix), np.inf)
            neg, pos = p < -1e-14, p > 1e-14
            ratios[neg] = -w[ix[neg]] / p[neg]
            ratios[pos] = (upper[ix[pos]] - w[ix[pos]]) / p[pos]
            k = ratios.argmin()
            step = np.clip(ratios[k], 0., 1.)
            w[ix] += step * p
            if step < 1. - 1e-12:
                j = ix[k]
                state[j] = 1 if p[k] > 0 else -1
                w[j] = upper[j] if p[k] > 0 else 0.
                continue
            g = h @ w + c
        else:
            lower = (state < 0) & ~lock
            capped = (state > 0) & ~lock
            lo = np.max(-g[lower], initial=-np.inf)
            hi = np.min(-g[capped], initial=np.inf)
            if lo > hi + 1e-11:
                state[np.flatnonzero(lower)[np.argmin(g[lower])]] = 0
                state[np.flatnonzero(capped)[np.argmax(g[capped])]] = 0
                continue
            nu = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.clip(0., lo, hi)
        shifted = g + nu
        bad = np.where(lock, 0., state * shifted)
        j = bad.argmax()
        if bad[j] <= 1e-11:
            break
        state[j] = 0
    else:
        raise RuntimeError('box-simplex active set did not converge')
    lam_lower = np.where((state < 0) | lock, np.maximum(shifted, 0.), 0.)
    lam_upper = np.where((state > 0) | lock, np.maximum(-shifted, 0.), 0.)
    return w, float(nu), lam_lower, lam_upper


def solve_batch(study: dict) -> dict:
    s = study
    correct = _calibration(s['sensor_calibration_sweeps'])
    y = correct(s['y_obs_raw'])
    response, source, nominal = _chamber(s)
    walls = _fit_walls(s, correct, response, source, nominal)
    design, fields = response(walls)
    z = np.asarray(s['Z_nuisance'], dtype=float)
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    yz = np.column_stack((design, y))
    projected = yz - z @ np.linalg.solve(gram, z.T @ yz)
    x, r = projected[:, :-1], projected[:, -1]
    ridge = s['source_ridge']
    h = x.T @ x + ridge * np.eye(x.shape[1])
    c = -x.T @ r
    upper = np.broadcast_to(s['u_box'], c.shape)
    budget = s['total_budget']
    w, nu, lower, cap = _box_simplex(h, c, upper, budget)
    err = x @ w - r
    objective = float((err @ err + ridge * (w @ w)) / 2)
    stationarity = h @ w + c + nu - lower + cap
    lagrangian = objective + nu * (w.sum() - budget) - lower @ w + cap @ (w - upper)
    dual = lagrangian - 0.5 * stationarity @ np.linalg.solve(h, stationarity)
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': cap,
                                'primal_objective': objective,
                                'dual_lower_bound': float(dual)}}
