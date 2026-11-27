import numpy as np
from itertools import combinations_with_replacement
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu


def _features(x, terms):
    return np.column_stack([np.prod(x[:, t], axis=1) for t in terms])


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], float)
    y = np.asarray(sweeps['raw_rosette'], float)
    terms = [()] + [t for degree in (1, 2, 3)
                    for t in combinations_with_replacement(range(4), degree)]
    features = _features(x, terms)
    delta = y - x
    folds = np.arange(len(x)) % 4
    scores = [np.mean(delta**2)]
    sizes = (0, 1, 5, 15, 35)
    for size in sizes[1:]:
        err = np.empty_like(delta)
        for fold in range(4):
            test = folds == fold
            p = features[:, :size]
            coef = np.linalg.lstsq(p[~test], delta[~test], rcond=None)[0]
            err[test] = delta[test] - p[test] @ coef
        scores.append(np.mean(err**2))
    size = sizes[int(np.argmin(scores))]
    terms = terms[:size]
    coef = np.linalg.lstsq(features[:, :size], delta, rcond=None)[0]
    deriv = [[(i, t.count(j), tuple(k for l, k in enumerate(t) if l != t.index(j)))
              for i, t in enumerate(terms) if j in t] for j in range(4)]

    def inverse(raw):
        raw = np.asarray(raw, float)
        if not size:
            return raw.copy()
        target = raw.reshape(-1, 4)
        z = target.copy()
        for _ in range(30):
            err = z + _features(z, terms) @ coef - target
            if np.max(np.abs(err)) < 1e-12:
                break
            J = np.broadcast_to(np.eye(4), (len(z), 4, 4)).copy()
            for j, row in enumerate(deriv):
                for i, count, t in row:
                    J[:, :, j] += (count * np.prod(z[:, t], axis=1))[:, None] * coef[i]
            step = np.linalg.solve(J, err[..., None])[..., 0]
            scale = 1.
            while scale > 1/1024:
                trial = z - scale * step
                trial_err = trial + _features(trial, terms) @ coef - target
                if np.linalg.norm(trial_err) < np.linalg.norm(err):
                    z = trial
                    break
                scale *= .5
            else:
                break
        return z.reshape(raw.shape)
    return inverse


def _axis(k, v, h):
    d = (k[:-1] + k[1:]) / (2*h*h)
    upper = d + np.maximum(-v[:-1], 0) / h
    lower = d + np.maximum(v[1:], 0) / h
    main = np.r_[upper, 0.] + np.r_[0., lower]
    return sparse.diags((-lower, main, -upper), (-1, 0, 1), format='csc')


def _transport(study, calibrate):
    nx, ny = study['nx'], study['ny']
    hx, hy = study['Lx']/(nx-1), study['Ly']/(ny-1)
    ax = _axis(np.asarray(study['kappa_x']), np.asarray(study['vel_x']), hx)
    ay = _axis(np.asarray(study['kappa_y']), np.asarray(study['vel_y']), hy)
    base = sparse.kronsum(ax, ay, format='csc')
    V = np.asarray(study['V_reaction_2d']).ravel(order='F')
    xx = np.tile(np.arange(nx), ny)
    yy = np.repeat(np.arange(ny), nx)
    walls = np.column_stack((xx == 0, xx == nx-1, yy == 0, yy == ny-1)).astype(float)
    d0 = np.repeat([study['robin_x']/hx, study['robin_y']/hy], 2)
    i, j = np.asarray(study['sensor_ij']).T
    ids = i + nx*j
    src = np.asarray(study['source_modes'], float).reshape(-1, nx*ny, order='F').T
    shifts = study['mu_shifts']

    def fields(d, rhs):
        diagonal = V + walls @ d
        for mu in shifts:
            yield splu(base + sparse.diags(diagonal + mu, format='csc')).solve(rhs)

    d = d0
    pulses = study['reference_tracer_pulses']
    if pulses:
        target = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
        forcing = src @ np.column_stack([p['w_pulse'] for p in pulses])

        def residual(d):
            return (np.concatenate([u[ids] for u in fields(d, forcing)]) - target).ravel()

        d = least_squares(residual, d0, bounds=(0, np.inf), gtol=1e-12).x
    observed = []
    for u in fields(d, src):
        observed.append(u[ids])
    return np.concatenate(observed), u.T.reshape(-1, nx, ny, order='F')


def _qp(S, y, ridge, upper, budget):
    n = S.shape[1]
    u = np.broadcast_to(np.asarray(upper, float), (n,))
    H = S.T @ S + ridge * np.eye(n)
    b = S.T @ y
    w = np.zeros(n)
    if budget >= u.sum():
        w = u.copy()
    elif budget > 0:
        w = u * (budget / u.sum())
        active = np.where(u == 0, -1, 0)
        for _ in range(20*n+100):
            free = active == 0
            g = H @ w - b
            if np.any(free):
                inv = solve(H[np.ix_(free, free)], np.column_stack((-g[free], np.ones(free.sum()))), assume_a='pos')
                nu = inv[:, 0].sum() / inv[:, 1].sum()
                p = inv[:, 0] - nu * inv[:, 1]
                if np.max(np.abs(p)) > 1e-12:
                    f = np.flatnonzero(free)
                    steps = np.full(len(f), np.inf)
                    neg, pos = p < -1e-14, p > 1e-14
                    steps[neg] = -w[f[neg]] / p[neg]
                    steps[pos] = (u[f[pos]] - w[f[pos]]) / p[pos]
                    j = int(np.argmin(steps))
                    if steps[j] < 1:
                        w[f] += max(0., steps[j]) * p
                        active[f[j]] = 1 if pos[j] else -1
                        w[f[j]] = u[f[j]] if pos[j] else 0.
                        continue
                    w[f] += p
                    g = H @ w - b
            else:
                lower, high = active < 0, active > 0
                nu = (-g[lower & (u > 0)].min() - g[high].max()) / 2
            violation = np.where(active > 0, g + nu, np.where((active < 0) & (u > 0), -g-nu, -np.inf))
            j = int(np.argmax(violation))
            if violation[j] <= 1e-11:
                break
            active[j] = 0
        else:
            raise RuntimeError('Box-simplex active set did not converge')
    g = H @ w - b
    lower, high = w <= 1e-11, w >= u - 1e-11
    free = ~(lower | high)
    if free.any():
        nu = -g[free].mean()
    else:
        lo = np.max(-g[lower & ~high], initial=-np.inf)
        hi = np.min(-g[high & ~lower], initial=np.inf)
        nu = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else lo if np.isfinite(lo) else hi if np.isfinite(hi) else 0.
    lam_lower = np.where(lower, np.maximum(g+nu, 0), 0.)
    lam_upper = np.where(high, np.maximum(-g-nu, 0), 0.)
    station = g + nu - lam_lower + lam_upper
    primal = .5 * (np.linalg.norm(S @ w - y)**2 + ridge * (w @ w))
    lagrangian = primal + nu*(w.sum()-budget) - lam_lower@w + lam_upper@(w-u)
    bound = lagrangian - .5 * station @ solve(H, station, assume_a='pos')
    return dict(weights=w, nu_sum=float(nu), lam_lower=lam_lower, lam_upper=lam_upper,
                primal_objective=float(primal), dual_lower_bound=float(bound))


def solve_batch(study: dict) -> dict:
    calibrate = _calibration(study['sensor_calibration_sweeps'])
    S, U_last = _transport(study, calibrate)
    Z = np.asarray(study['Z_nuisance'], float)
    y = calibrate(study['y_obs_raw'])
    T = np.column_stack((S, y))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    T -= Z @ solve(gram, Z.T @ T, assume_a='pos')
    cert = _qp(T[:, :-1], T[:, -1], study['source_ridge'], study['u_box'], study['total_budget'])
    w = cert['weights']
    return dict(weights=w, terminal_field=np.tensordot(w, U_last, axes=(0, 0)), kkt_certificate=cert)
