"""Periodic transport inversion with empirical rosette calibration and a KKT certificate."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    # Extend the interior finite-volume stencil across the periodic seam.
    scale = 0.5 / h**2
    left = -(kappa + np.roll(kappa, 1)) * scale - np.maximum(velocity, 0) / h
    right = -(kappa + np.roll(kappa, -1)) * scale - np.maximum(-velocity, 0) / h
    operator = diags((left[1:], -left - right, right[:-1]), (-1, 0, 1), format='lil')
    operator[0, -1] += left[0]
    operator[-1, 0] += right[-1]
    return operator.tocsc()

def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    design = np.column_stack((np.ones(len(ref)), ref))
    coefficients = np.linalg.lstsq(design, sweeps['raw_rosette'], rcond=None)[0]
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coefficients[1:].T, (raw - coefficients[0]).T).T.ravel()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    axes = [_axis(np.asarray(study['kappa_' + k]), np.asarray(study['vel_' + k]),
                  study['L' + k] / (n - 1))
            for k, n in (('x', nx), ('y', ny))]
    operator = (kron(eye(ny), axes[0], format='csc')
                + kron(axes[1], eye(nx), format='csc')
                + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    sources = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ix, iy = np.asarray(study['sensor_ij']).T
    sensor = ix + nx * iy
    rows = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(sources)
        rows.append(field[sensor, :])
    return np.vstack(rows), field


def _box_budget(H, c, cap, budget):
    n = len(c)
    if budget == 0 or cap == 0:
        return np.zeros(n), float(min(c))
    if budget >= n * cap:
        w = np.full(n, cap)
        return w, float(max(H @ w + c))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1 lower, +1 upper
    tol = 5e-13 * max(1., np.max(np.abs(c)), cap * np.max(np.sum(abs(H), axis=1)))
    for _ in range(30 * n + 30):
        free = np.flatnonzero(active == 0)
        g = H @ w + c
        if not len(free):
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            active[[i, j]] = 0
            continue
        fixed = np.flatnonzero(active)
        m = len(free)
        system = np.ones((m + 1, m + 1))
        system[:m, :m] = H[np.ix_(free, free)]
        system[m, m] = 0
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        sol = np.linalg.solve(system, rhs)
        candidate, nu = sol[:-1], -float(sol[-1])
        direction = candidate - w[free]
        ratio = np.full(m, np.inf)
        low, high = (candidate < 0) & (direction < 0), (candidate > cap) & (direction > 0)
        ratio[low] = -w[free[low]] / direction[low]
        ratio[high] = (cap - w[free[high]]) / direction[high]
        k = np.argmin(ratio)
        if ratio[k] < 1:
            w[free] += max(0., ratio[k]) * direction
            j = free[k]
            active[j] = -1 if low[k] else 1
            w[j] = 0. if low[k] else cap
            continue
        w[free] = np.clip(candidate, 0., cap)
        violation = active * (H @ w + c - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, nu
        active[j] = 0
    raise RuntimeError('Box-and-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    y = _calibrate(study)
    S, last = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    observations = np.column_stack((S, y))
    profiled = observations - Z @ np.linalg.solve(
        Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ observations)
    D, target = profiled[:, :-1], profiled[:, -1]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = D.T @ D + ridge * np.eye(D.shape[1])
    w, nu = _box_budget(H, -D.T @ target, cap, budget)
    residual = D @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = D.T @ residual + ridge * w
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(0, reduced).sum()
    return {
        'weights': w,
        'terminal_field': (last @ w).reshape((study['nx'], study['ny']), order='F'),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0, reduced), 'lam_upper': np.maximum(0, -reduced),
            'primal_objective': objective, 'dual_lower_bound': float(dual),
        },
    }
