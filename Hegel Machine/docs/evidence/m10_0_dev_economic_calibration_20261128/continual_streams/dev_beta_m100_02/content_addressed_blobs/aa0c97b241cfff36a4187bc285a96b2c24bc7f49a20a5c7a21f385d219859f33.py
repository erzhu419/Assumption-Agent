"""Periodic-chamber transport inversion with empirical rosette calibration."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

def _periodic_axis(kappa, velocity, h):
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
    # Held-out chamber measurements favor periodic closure on both axes.
    ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _periodic_axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
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


def _cross_design(b, f):
    b = np.asarray(b)
    return np.stack((b, b*b, np.sin(f*b), 1-np.cos(f*b)), axis=-1)


def _sweep_fit(study, use=None):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = (np.asarray(sweeps[key]) for key in ('ref_rosette', 'raw_rosette'))
    single = sweeps['single_bank_sweeps']
    ids, x, y = (np.asarray(single[key]) for key in ('bank_id', 'ref_signal', 'raw_readout'))
    clean = (abs(raw) < 5).all(axis=1)
    if use is not None:
        clean &= use
    ref, raw = ref[clean], raw[clean]
    a = np.r_[ref.ravel(), x]
    b = np.r_[np.roll(ref, 2, axis=1).ravel(), np.zeros(len(x))]
    yy = np.r_[raw.ravel(), y]
    bank = np.r_[np.tile(np.arange(4), len(ref)), ids].astype(int)
    is_single = (b == 0) & (abs(yy) < 5)
    def isolated_loss(freq):
        z = a[is_single]
        e = np.eye(4)[bank[is_single]]
        D = np.column_stack((e, e*z[:,None], e*np.sin(freq*z[:,None])))
        t = yy[is_single]
        return np.linalg.norm(D @ np.linalg.lstsq(D, t, rcond=None)[0] - t)
    f = minimize_scalar(isolated_loss, bounds=(1.5, 3.5), method='bounded').x
    keep = abs(yy) < 5
    a, b, yy, bank = a[keep], b[keep], yy[keep], bank[keep]
    diag = np.column_stack((np.ones(len(a)), a, np.sin(f*a)))
    coeff = np.array([np.linalg.lstsq(diag[(bank==j)&(b==0)], yy[(bank==j)&(b==0)], rcond=None)[0] for j in range(4)])
    # Profile the cross-bank frequency separately from the self-response.
    def profile(freq):
        cross = _cross_design(b, freq)
        local = coeff.copy()
        for _ in range(5):
            h = np.linalg.lstsq(cross*local[bank,1,None],
                                yy-np.sum(diag*local[bank], axis=1), rcond=None)[0]
            D = diag.copy()
            D[:,1] += cross @ h
            local = np.array([np.linalg.lstsq(D[bank==j], yy[bank==j], rcond=None)[0]
                              for j in range(4)])
        return np.linalg.norm(np.sum(D*local[bank], axis=1)-yy), local, h
    cross_f = minimize_scalar(lambda freq: profile(freq)[0], bounds=(1.5, 2.5),
                              method='bounded').x
    _, coeff, h = profile(cross_f)
    return f, cross_f, coeff, h


def _raw(x, fit):
    f, cross_f, coeff, h = fit
    b = np.roll(x, 2, axis=-1)
    return coeff[:,0] + coeff[:,1]*(x + _cross_design(b, cross_f) @ h) + coeff[:,2]*np.sin(f*x)


def _calibrate(study):
    f, cross_f, c, h = fit = _sweep_fit(study)
    y = np.asarray(study['y_obs_raw']).reshape(-1,4)
    z = (y-c[:,0])/(c[:,1]+f*c[:,2])
    for _ in range(30):
        r = _raw(z, fit)-y
        if np.max(abs(r)) < 1e-12:
            break
        b = np.roll(z, 2, axis=-1)
        d0 = c[:,1]+f*c[:,2]*np.cos(f*z)
        d2 = c[:,1]*(h[0] + 2*h[1]*b + cross_f*(h[2]*np.cos(cross_f*b)+h[3]*np.sin(cross_f*b)))
        roll = lambda q: np.roll(q, 2, axis=-1)
        dz = (r*roll(d0) - d2*roll(r))/(d0*roll(d0)-d2*roll(d2))
        z -= np.clip(dz, -.35, .35)
    return z.ravel()


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
