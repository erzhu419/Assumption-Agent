from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _sweep_basis(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=2)


def _calibrator(sweep):
    x = np.asarray(sweep['ref_rosette'], dtype=float)
    y = np.asarray(sweep['raw_rosette'], dtype=float)
    bank = sweep.get('single_bank_sweeps')
    if bank is not None:
        ids = np.asarray(bank['bank_id'], dtype=int)
        z = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
        target = np.full_like(z, np.nan)
        target[np.arange(len(ids)), ids] = bank['raw_readout']
        x, y = np.vstack((x, z)), np.vstack((y, target))
    units = np.eye(4, dtype=int)
    linear = np.vstack((np.zeros((1, 4), dtype=int), units))
    quad = np.array([np.bincount(k, minlength=4) for k in combinations_with_replacement(range(4), 2)])
    cubic = np.array([np.bincount(k, minlength=4) for k in combinations_with_replacement(range(4), 3)])
    candidates = [linear, np.vstack((linear, 2*units)), np.vstack((linear, quad)),
                  np.vstack((linear, 2*units, 3*units)), np.vstack((linear, quad, 3*units)),
                  np.vstack((linear, quad, cubic))]
    valid = np.isfinite(y)
    fold_ids = np.arange(len(x)) % 4
    best = np.inf
    powers = linear
    for candidate in candidates:
        a = _sweep_basis(x, candidate)
        pred = np.full_like(y, np.nan)
        identified = True
        for fold in range(4):
            test = fold_ids == fold
            for j in range(4):
                train = ~test & valid[:, j]
                coef, _, rank, singular = np.linalg.lstsq(a[train], y[train, j], rcond=None)
                if rank < a.shape[1] or singular[-1] < 1e-6 * singular[0]:
                    identified = False
                    break
                pred[test, j] = a[test] @ coef
            if not identified:
                break
        if identified:
            err = np.sqrt(np.mean((pred[valid] - y[valid])**2))
            if err < .85 * best:
                powers, best = candidate, err
    a = _sweep_basis(x, powers)
    coef = np.column_stack([np.linalg.lstsq(a[mask], y[mask, j], rcond=None)[0]
                            for j, mask in enumerate(valid.T)])
    jac_powers = np.maximum(powers[None, :, :] - units[:, None, :], 0)
    jac_coef = powers.T[:, :, None] * coef[None, :, :]

    def calibrate(raw):
        yraw = np.asarray(raw).reshape(-1, 4)
        estimate = np.linalg.solve(coef[1:5].T, (yraw - coef[0]).T).T
        if len(powers) == 5:
            return estimate.ravel()

        def jac(z):
            dz = np.prod(z[None, None, :] ** jac_powers, axis=2)
            return np.einsum('im,imj->ji', dz, jac_coef)

        return np.vstack([least_squares(lambda z: (_sweep_basis(z[None], powers) @ coef)[0] - row,
                                       start, jac=jac, gtol=1e-12).x
                          for row, start in zip(yraw, estimate)]).ravel()
    return calibrate


def _axis(k, velocity, h):
    k = np.asarray(k)
    v = np.asarray(velocity) / h
    diff = (k[:-1] + k[1:]) / (2*h*h)
    lower = diff + np.maximum(v[1:], 0)
    upper = diff + np.maximum(-v[:-1], 0)
    return diags((-lower, np.r_[upper, 0.] + np.r_[0., lower], -upper), (-1, 0, 1), format='csc')


class _Transport:
    def __init__(self, s):
        nx, ny = s['nx'], s['ny']
        self.shape = nx, ny
        hx, hy = s['Lx']/(nx-1), s['Ly']/(ny-1)
        self.nominal = np.repeat([s['robin_x']/hx, s['robin_y']/hy], 2)
        self.identity = eye(nx*ny, format='csc')
        self.matrix = (kron(eye(ny), _axis(s['kappa_x'], s['vel_x'], hx))
                       + kron(_axis(s['kappa_y'], s['vel_y'], hy), eye(nx))
                       + diags(np.asarray(s['V_reaction_2d']).ravel(order='F'))).tocsc()
        ix, iy = np.asarray(s['sensor_ij']).T
        self.sensors = ix + nx*iy
        x = np.tile(np.arange(nx), ny)
        y = np.repeat(np.arange(ny), nx)
        self.walls = np.column_stack([x == 0, x == nx-1, y == 0, y == ny-1]).astype(float)
        self.modes = np.column_stack([a.ravel(order='F') for a in np.asarray(s['source_modes'])])
        self.shifts = s['mu_shifts']

    def response(self, wall, rhs):
        a = self.matrix + diags(self.walls @ wall, format='csc')
        observed = []
        for shift in self.shifts:
            field = splu(a + shift*self.identity).solve(rhs)
            observed.append(field[self.sensors])
        return np.concatenate(observed), field


def _wall_fit(model, pulses, calibrate):
    if not pulses:
        return model.nominal
    rhs = model.modes @ np.column_stack([p['w_pulse'] for p in pulses])
    data = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    fit = least_squares(lambda wall: (model.response(wall, rhs)[0] - data).ravel(),
                        np.maximum(model.nominal, 1e-8), bounds=(0, np.inf), gtol=1e-12)
    return fit.x


def _box_qp(h, q, u, budget):
    n = len(q)
    w = u * (budget / u.sum()) if u.sum() else np.zeros(n)
    status = np.zeros(n, dtype=int)
    status[u == 0] = -1
    tol = 2e-12
    for _ in range(20*n*n+1):
        free = status == 0
        g = h @ w + q
        p = np.zeros(n)
        if free.any():
            f = h[np.ix_(free, free)]
            z, v = np.linalg.solve(f, np.column_stack((-g[free], np.ones(f.shape[0])))).T
            nu = z.sum()/v.sum()
            p[free] = z - nu*v
        else:
            lower = (status < 0) & (u > 0)
            upper = status > 0
            lo = np.max(-g[lower], initial=-np.inf)
            hi = np.min(-g[upper], initial=np.inf)
            nu = np.clip(0., lo, hi) if lo <= hi else (lo+hi)/2
        if np.max(np.abs(p)) <= tol:
            violation = status*(g+nu)
            violation[u == 0] = 0.
            j = np.argmax(violation)
            if violation[j] <= tol:
                return w, nu
            status[j] = 0
            continue
        step = np.full(n, np.inf)
        up = p > tol
        down = p < -tol
        step[up] = (u[up]-w[up])/p[up]
        step[down] = -w[down]/p[down]
        j = np.argmin(step)
        alpha = np.clip(step[j], 0., 1.)
        w += alpha*p
        if alpha < 1.:
            status[j] = 1 if up[j] else -1
            w[j] = u[j] if up[j] else 0.
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    calibrate = _calibrator(s['sensor_calibration_sweeps'])
    model = _Transport(s)
    wall = _wall_fit(model, s.get('reference_tracer_pulses'), calibrate)
    S, U_last = model.response(wall, model.modes)
    y = calibrate(s['y_obs_raw'])
    Z = np.asarray(s['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (Pz @ S)
    y = y - Z @ (Pz @ y)
    ridge = s['source_ridge']
    h = S.T @ S + ridge*np.eye(S.shape[1])
    q = -S.T @ y
    u = np.broadcast_to(s['u_box'], q.shape)
    budget = s['total_budget']
    w, nu = _box_qp(h, q, u, budget)
    g = h @ w + q + nu
    lower = np.where(w <= 1e-10, np.maximum(g, 0.), 0.)
    upper = np.where(w >= u-1e-10, np.maximum(-g, 0.), 0.)
    stat = g - lower + upper
    residual = S @ w - y
    obj = .5*(residual @ residual + ridge*(w @ w))
    dual = obj + nu*(w.sum()-budget) - lower @ w + upper @ (w-u) - .5*stat @ np.linalg.solve(h, stat)
    return {'weights': w, 'terminal_field': (U_last @ w).reshape(model.shape, order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': float(nu), 'lam_lower': lower,
                                'lam_upper': upper, 'primal_objective': float(obj),
                                'dual_lower_bound': float(dual)}}
