"""Periodic-x, reflecting-y transport and certified source inversion."""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kap, vel, h, periodic):
    faces = (kap[:-1] + kap[1:]) / (2 * h**2)
    lower = -faces - np.maximum(vel[1:], 0) / h
    upper = -faces - np.maximum(-vel[:-1], 0) / h
    wrap_lower = wrap_upper = 0.
    if periodic:
        face = (kap[0] + kap[-1]) / (2 * h**2)
        wrap_lower = -face - max(vel[0], 0) / h
        wrap_upper = -face - max(-vel[-1], 0) / h
    else:
        upper[0] = -kap[0] / h**2
        lower[-1] = -kap[-1] / h**2
    main = -np.r_[wrap_lower, lower] - np.r_[upper, wrap_upper]
    mat = diags((lower, main, upper), (-1, 0, 1), format='lil')
    if periodic:
        mat[0, -1] += wrap_lower
        mat[-1, 0] += wrap_upper
    return mat.tocsc()


def _responses(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = (_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (study['n' + a] - 1), a == 'x')
              for a in ('x', 'y'))
    operator = (kron(eye(ny, format='csc'), ax, format='csc')
                + kron(ay, eye(nx, format='csc'), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    sensor_i, sensor_j = study['sensor_ij'].T
    rows = sensor_i + nx * sensor_j
    ident = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(rhs)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


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


def solve_downstream(cal_item: dict) -> dict:
    """Fit weights from observations already in canonical rosette coordinates."""
    S, U_last = _responses(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    data = np.column_stack((S, cal_item['y_obs_raw']))
    data -= Z @ np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = cal_item['source_ridge']
    upper, total = cal_item['u_box'], cal_item['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(len(U_last)), -S.T @ y, upper, total)
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    dual = float(objective - g @ w + nu * total + upper * np.minimum(0., reduced).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., reduced).tolist(),
                                'lam_upper': np.maximum(0., -reduced).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
