"""Calibrated steady transport inversion on a box simplex."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibrator(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    x = (x - mean) / scale
    powers = [()] + [(i,) for i in range(4)]
    candidates = [powers[:]]
    for degree in (2, 3):
        powers += list(combinations_with_replacement(range(4), degree))
        if len(powers) <= len(x) * 0.7:
            candidates.append(powers[:])

    def design(z, terms):
        return np.column_stack([np.prod(z[:, t], axis=1) for t in terms])

    best = np.inf
    for terms in candidates:
        D = design(x, terms)
        sq = 0.0
        for k in range(5):
            hold = np.arange(len(x)) % 5 == k
            coef = np.linalg.lstsq(D[~hold], raw[~hold], rcond=None)[0]
            sq += np.sum((D[hold] @ coef - raw[hold]) ** 2)
        if sq < 0.85 * best:
            best = sq
            selected = terms
    coef = np.linalg.lstsq(design(x, selected), raw, rcond=None)[0]
    linear = np.linalg.lstsq(design(x, candidates[0]), raw, rcond=None)[0]

    def inverse(y):
        y = np.asarray(y, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(linear[1:].T, (y - linear[0]).T).T
        if len(selected) > 5:
            for i, target in enumerate(y):
                def residual(p):
                    return design(p[None, :], selected)[0] @ coef - target

                def jacobian(p):
                    jac = np.zeros((4, 4))
                    for term, c in zip(selected[1:], coef[1:]):
                        for j, channel in enumerate(term):
                            jac[:, channel] += c * np.prod(p[list(term[:j] + term[j+1:])])
                    return jac

                z[i] = least_squares(residual, z[i], jac=jacobian,
                                     gtol=1e-11, max_nfev=40).x
        return (z * scale + mean).ravel()

    return inverse


def _axis(k, v, h):
    k, v = np.asarray(k), np.asarray(v) / h
    f = (k[:-1] + k[1:]) / (2 * h**2)
    lower = -f - np.maximum(v[1:], 0)
    upper = -f + np.minimum(v[:-1], 0)
    return diags((lower, np.r_[-upper, 0] + np.r_[0, -lower], upper),
                 (-1, 0, 1), format='csc')


def _transport(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx'] / (nx - 1), s['Ly'] / (ny - 1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    identity = eye(nx * ny, format='csc')
    base = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
            diags(np.asarray(s['V_reaction_2d']).ravel(order='F'))).tocsc()
    B = np.column_stack([np.asarray(src).ravel(order='F') for src in s['source_modes']])
    i, j = np.asarray(s['sensor_ij']).T
    ix = i + nx * j
    walls = np.column_stack([np.tile(np.arange(nx) == k, ny) for k in (0, nx-1)] +
                            [np.repeat(np.arange(ny) == k, nx) for k in (0, ny-1)])
    nominal = np.repeat([s['robin_x'] / hx, s['robin_y'] / hy], 2)

    def forward(wall, rhs=B):
        A = base + diags(walls @ wall, format='csc')
        readings = []
        for mu in s['mu_shifts']:
            U = splu(A + mu * identity).solve(rhs)
            readings.append(U[ix])
        return np.concatenate(readings, axis=0), U

    return forward, B, nominal


def _box_simplex(H, c, u, budget):
    """Feasible active-set QP. Status is -1 at lower, +1 at upper, 0 free."""
    n = len(c)
    x = u * (budget / u.sum()) if u.sum() else np.zeros(n)
    status = np.zeros(n, dtype=int)
    status[u == 0] = -1
    scale = max(1.0, np.max(np.abs(c)), np.max(np.abs(H)) * max(1., budget))
    tol = 1e-12 * scale
    for _ in range(20 * (n + 1)**2):
        free = status == 0
        g = H @ x - c
        if not free.any():
            lo = (status == -1) & (u > 0)
            hi = status == 1
            a = np.max(-g[lo], initial=-np.inf)
            b = np.min(-g[hi], initial=np.inf)
            if a <= b + tol:
                nu = np.clip(0., a, b)
                return x, nu
            status[np.flatnonzero(lo)[np.argmax(-g[lo])]] = 0
            status[np.flatnonzero(hi)[np.argmin(-g[hi])]] = 0
            continue
        inds = np.flatnonzero(free)
        k = len(inds)
        F = H[np.ix_(inds, inds)]
        K = np.block([[F, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
        rhs = np.r_[-g[free], budget - x.sum()]
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        p, nu = sol[:-1], sol[-1]
        trial = x[free] + p
        ratio = np.full(k, np.inf)
        low = trial < -1e-13
        high = trial > u[free] + 1e-13
        ratio[low] = -x[free][low] / p[low]
        ratio[high] = (u[free][high] - x[free][high]) / p[high]
        hit = np.argmin(ratio)
        if ratio[hit] < 1:
            x[free] += max(0., ratio[hit]) * p
            i = inds[hit]
            status[i] = 1 if high[hit] else -1
            x[i] = u[i] if high[hit] else 0.
            continue
        x[free] = np.clip(trial, 0., u[free])
        g = H @ x - c + nu
        violation = np.where(u > 0, status * g, 0.)
        worst = np.argmax(violation)
        if violation[worst] <= tol:
            return x, nu
        status[worst] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    calibrate = _calibrator(s['sensor_calibration_sweeps'])
    y = calibrate(s['y_obs_raw'])
    forward, B, nominal = _transport(s)
    m = B.shape[1]
    u = np.broadcast_to(np.asarray(s['u_box'], dtype=float), (m,))
    budget = float(s['total_budget'])
    ridge = float(s['source_ridge'])
    Z = np.asarray(s['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    y_tilde = y - Z @ (Pz @ y)

    def fit(S):
        X = S - Z @ (Pz @ S)
        H = X.T @ X + ridge * np.eye(m)
        c = X.T @ y_tilde
        w, nu = _box_simplex(H, c, u, budget)
        return X, H, c, w, nu

    pulses = s['reference_tracer_pulses']
    if pulses:
        W = np.column_stack([p['w_pulse'] for p in pulses])
        Y = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
        rhs = B @ W
        wall = least_squares(lambda b: (forward(b, rhs)[0] - Y).ravel(),
                             nominal, bounds=(0., np.inf), gtol=1e-12).x
    else:
        # Blind observations confound wall and source effects. Anchor the wall
        # correction to the supplied coefficients on a relative scale.
        S0, _ = forward(nominal)
        anchor = max(1e-8, 0.2 * np.linalg.norm(S0 @ fit(S0)[3]))

        def residual(delta):
            S, _ = forward(nominal * np.exp(delta))
            X, _, _, w, _ = fit(S)
            return np.r_[X @ w - y_tilde, anchor * delta]

        delta = least_squares(residual, np.zeros(4), bounds=(-0.7, 0.7),
                              gtol=1e-9).x
        wall = nominal * np.exp(delta)

    S, U = forward(wall)
    X, H, c, w, nu = fit(S)
    g = H @ w - c + nu
    lower = np.where(w <= 1e-12, np.maximum(g, 0.), 0.)
    upper = np.where(u - w <= 1e-12, np.maximum(-g, 0.), 0.)
    residual = X @ w - y_tilde
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lagrangian = primal + nu * (w.sum() - budget) - lower @ w + upper @ (w - u)
    stationarity = g - lower + upper
    if ridge > 0.:
        dual = lagrangian - 0.5 * stationarity @ np.linalg.solve(H, stationarity)
    else:
        # Supporting hyperplane of the equality Lagrangian on the box.
        dual = (primal + nu * (w.sum() - budget) +
                np.minimum(-g*w, g*(u-w)).sum())
    return {'weights': w, 'terminal_field': (U @ w).reshape((s['nx'], s['ny']), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': float(nu),
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': primal,
                                'dual_lower_bound': float(min(primal, dual))}}
