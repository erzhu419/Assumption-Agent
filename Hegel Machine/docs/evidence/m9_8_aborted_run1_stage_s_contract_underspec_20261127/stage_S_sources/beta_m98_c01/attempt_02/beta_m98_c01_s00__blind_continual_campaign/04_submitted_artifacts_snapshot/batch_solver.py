"""Calibrate the rosettes, reconstruct the chamber, and certify the source QP."""
import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _monomials(degree):
    return np.array([[idx.count(j) for j in range(4)]
                     for d in range(degree + 1)
                     for idx in itertools.combinations_with_replacement(range(4), d)])


def _features(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=-1)


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    # Score entire rosettes rather than splitting their four coupled channels.
    fold = np.arange(len(x)) % 4
    models = []
    for degree in (1, 2, 3):
        powers = _monomials(degree)
        a = _features(x, powers)
        if len(a) * .75 <= a.shape[1] + 4:
            break
        scores = []
        for k in range(4):
            tr = fold != k
            b = np.linalg.lstsq(a[tr], (y-x)[tr], rcond=None)[0]
            scores.append(np.mean((a[~tr] @ b - (y-x)[~tr]) ** 2))
        models.append((np.mean(scores), np.std(scores) / 2, powers, a))
    best = min(models, key=lambda m: m[0])
    _, _, powers, a = next(m for m in models if m[0] <= 1.10 * best[0])
    b = np.linalg.lstsq(a, y-x, rcond=None)[0]
    linear = (np.eye(4) + b[1:5]).T
    derivatives = np.maximum(powers[None, :, :] - np.eye(4, dtype=int)[:, None, :], 0)

    def invert(raw):
        raw = np.asarray(raw, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(linear, (raw - b[0]).T).T
        if len(powers) == 5:
            return z.ravel()
        for i, target in enumerate(raw):
            def fun(t):
                return t + _features(t[None, :], powers)[0] @ b - target

            def jac(t):
                d = np.prod(t[None, None, :] ** derivatives, axis=-1) * powers.T
                return np.eye(4) + (d @ b).T

            z[i] = least_squares(fun, z[i], jac=jac, gtol=1e-12, ftol=1e-12,
                                 xtol=1e-12).x
        return z.ravel()

    return invert


def _axis(kappa, velocity, step):
    d = (np.asarray(kappa[:-1]) + kappa[1:]) / (2 * step**2)
    a = np.asarray(velocity) / step
    lower = -d - np.maximum(a[1:], 0)
    upper = -d + np.minimum(a[:-1], 0)
    center = -np.r_[upper, 0.] - np.r_[0., lower]
    return diags((lower, center, upper), (-1, 0, 1), format='csc')


def _chamber(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx']/(nx-1), s['Ly']/(ny-1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    ident = eye(nx*ny, format='csc')
    base = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
            + diags(np.ravel(s['V_reaction_2d'], order='F'), format='csc'))
    xx, yy = np.tile(np.arange(nx), ny), np.repeat(np.arange(ny), nx)
    walls = np.column_stack((xx == 0, xx == nx-1, yy == 0, yy == ny-1)).astype(float)
    walls /= [hx, hx, hy, hy]
    si = np.asarray(s['sensor_ij'])
    indices = si[:, 0] + nx*si[:, 1]
    sources = np.column_stack([m.ravel(order='F') for m in s['source_modes']])

    def forward(robin, rhs):
        operator = base + diags(walls @ robin, format='csc')
        readings = []
        for shift in s['mu_shifts']:
            u = splu(operator + shift*ident).solve(rhs)
            readings.append(u[indices])
        return np.concatenate(readings), u

    nominal = np.repeat([s['robin_x'], s['robin_y']], 2).astype(float)
    return forward, nominal, sources


def _pulse_fit(pulses, invert, forward, nominal, sources):
    rhs = sources @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([invert(p['y_pulse_raw']) for p in pulses])
    fit = least_squares(lambda r: (forward(r, rhs)[0] - observed).ravel(), nominal,
                        bounds=(0., np.inf), ftol=1e-12, gtol=1e-12, xtol=1e-12)
    return fit.x


def _box_qp(h, c, upper, budget):
    """Feasible active set for a convex quadratic with one equality."""
    n = len(c)
    w = budget * upper / upper.sum() if upper.sum() else np.zeros(n)
    g = lambda: h @ w + c
    if budget == 0:
        return w, float(-min(g()))
    if budget == upper.sum():
        return upper.copy(), float(-max(h @ upper + c))
    status = np.where(upper > 0, 0, 2)
    tol = 5e-13
    for _ in range(20*n*n):
        free = np.flatnonzero(status == 0)
        grad = g()
        if not len(free):
            lower, high = np.where(status == -1)[0], np.where(status == 1)[0]
            i, j = lower[np.argmin(grad[lower])], high[np.argmax(grad[high])]
            if grad[j] <= grad[i] + tol:
                return w, float(-(grad[i] + grad[j])/2)
            status[[i, j]] = 0
            continue
        f = h[np.ix_(free, free)]
        v, d = np.linalg.solve(f, np.column_stack((-grad[free], np.ones(len(free))))).T
        nu = v.sum()/d.sum()
        p = v - nu*d
        alpha = np.ones(len(free))
        positive, negative = p > tol, p < -tol
        alpha[positive] = (upper[free[positive]] - w[free[positive]]) / p[positive]
        alpha[negative] = -w[free[negative]] / p[negative]
        j = np.argmin(alpha)
        if alpha[j] < 1:
            w[free] += max(0., alpha[j])*p
            i = free[j]
            status[i] = 1 if p[j] > 0 else -1
            w[i] = upper[i] if p[j] > 0 else 0.
            continue
        w[free] += p
        reduced = g() + nu
        violations = np.where(status == -1, -reduced, np.where(status == 1, reduced, 0.))
        i = np.argmax(violations)
        if violations[i] <= tol:
            return w, float(nu)
        status[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    invert = _calibration(s['sensor_calibration_sweeps'])
    y = invert(s['y_obs_raw'])
    forward, robin, sources = _chamber(s)
    pulses = s.get('reference_tracer_pulses')
    if pulses:
        robin = _pulse_fit(pulses, invert, forward, robin, sources)
    design, last = forward(robin, sources)
    z = np.asarray(s['Z_nuisance'])
    a = np.column_stack((design, y))
    a -= z @ np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T @ a)
    design, y = a[:, :-1], a[:, -1]
    ridge = s['source_ridge']
    h = design.T @ design + ridge*np.eye(design.shape[1])
    c = -design.T @ y
    budget = float(s['total_budget'])
    upper = np.broadcast_to(s['u_box'], c.shape).astype(float)
    w, nu = _box_qp(h, c, upper, budget)
    reduced = h @ w + c + nu
    lo = np.where(w <= 1e-11, np.maximum(reduced, 0.), 0.)
    hi = np.where(w >= upper-1e-11, np.maximum(-reduced, 0.), 0.)
    residual = design @ w - y
    obj = .5*(residual @ residual + ridge*(w @ w))
    # Minimize the Lagrangian over unconstrained weights to get a dual bound.
    lag_grad = reduced - lo + hi
    dual = (obj + nu*(w.sum()-budget) - lo @ w + hi @ (w-upper)
            - .5*lag_grad @ np.linalg.solve(h, lag_grad))
    return {'weights': w, 'terminal_field': (last @ w).reshape((s['nx'], s['ny']), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': lo,
                                'lam_upper': hi, 'primal_objective': float(obj),
                                'dual_lower_bound': float(min(obj, dual))}}
