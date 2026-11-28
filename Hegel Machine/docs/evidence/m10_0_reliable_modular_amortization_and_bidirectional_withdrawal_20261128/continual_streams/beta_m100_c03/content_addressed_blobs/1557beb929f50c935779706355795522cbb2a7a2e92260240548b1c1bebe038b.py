"""Calibrated chamber inversion with a periodic x seam and reflecting y walls."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h, periodic):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    n = len(kappa)
    face = (kappa[:-1] + kappa[1:]) / (2*h**2)
    if periodic:
        seam = (kappa[0] + kappa[-1]) / (2*h**2)
        left = np.r_[seam, face] + np.maximum(vel, 0) / h
        right = np.r_[face, seam] + np.maximum(-vel, 0) / h
        axis = diags((-left[1:], left + right, -right[:-1]), (-1, 0, 1), format='lil')
        axis[0, n-1] -= left[0]
        axis[-1, 0] -= right[-1]
        return axis.tocsc()
    left = np.r_[face[:-1] + np.maximum(vel[1:-1], 0) / h, kappa[-1] / h**2]
    right = np.r_[kappa[0] / h**2, face[1:] + np.maximum(-vel[1:-1], 0) / h]
    main = np.r_[right[0], left[:-1] + right[1:], left[-1]]
    return diags((-left, main, -right), (-1, 0, 1), format='csc')


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + d], study['vel_' + d], study['L' + d] / (n-1), d == 'x')
              for d, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    ident = eye(nx*ny, format='csc')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T.copy()
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx*sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(shifts) > 1 and len(indices) == np.size(study['y_obs_raw']):
        indices = indices.reshape(len(shifts), -1)
    sampled = []
    for k, mu in enumerate(shifts):
        fields = splu(operator + mu*ident).solve(rhs)
        sampled.append(fields[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(sampled), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, budget):
    """Primal active set for a strictly convex quadratic with a fixed sum."""
    n = len(c)
    if budget <= 0:
        return np.zeros(n), float(np.min(c))
    if budget >= n*cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w + c))
    w = np.full(n, budget/n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    xtol = 64*np.finfo(float).eps * cap * n
    gtol = 128*np.finfo(float).eps * max(1., np.linalg.norm(c, np.inf), cap*np.linalg.norm(H, np.inf))
    for _ in range(30*n + 100):
        free = np.flatnonzero(state == 0)
        g = H @ w + c
        if not len(free):
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[i] >= g[j] - gtol:
                return w, float((g[i] + g[j])/2)
            state[[i, j]] = 0
            continue
        fixed = np.flatnonzero(state != 0)
        target = budget - np.sum(w[fixed])
        a, b = np.linalg.solve(H[np.ix_(free, free)], np.column_stack((
            -c[free] - H[np.ix_(free, fixed)] @ w[fixed], np.ones(len(free))))).T
        nu = (target - np.sum(a))/np.sum(b)
        z = a + nu*b
        delta = z - w[free]
        alpha = np.ones(len(free))
        lo, hi = z < -xtol, z > cap + xtol
        alpha[lo] = -w[free[lo]]/delta[lo]
        alpha[hi] = (cap-w[free[hi]])/delta[hi]
        if np.any(lo | hi):
            hit = np.argmin(alpha)
            w[free] += min(1., max(0., alpha[hit]))*delta
            i = free[hit]
            state[i] = -1 if lo[hit] else 1
            w[i] = 0. if lo[hit] else cap
            continue
        w[free] = np.clip(z, 0., cap)
        g = H @ w + c
        nu = float(np.mean(g[free]))
        violation = state*(g-nu)
        i = np.argmax(violation)
        if violation[i] <= gtol:
            return w, nu
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    S, U_last = _responses(study)
    y = np.asarray(study['y_obs_raw']).ravel()
    Z = np.asarray(study['Z_nuisance'])
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
    data = np.column_stack((S, y))
    data -= Z @ np.linalg.solve(Z.T @ Z + study['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ data)
    design, obs = data[:, :-1], data[:, -1]
    H = design.T @ design + ridge*np.eye(design.shape[1])
    w, nu = _box_simplex(H, -design.T @ obs, cap, budget)
    residual = design @ w - obs
    g = design.T @ residual + ridge*w
    J = float((residual @ residual + ridge*(w @ w))/2)
    d = g-nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': J,
                                'dual_lower_bound': float(J - g @ w + nu*budget + cap*np.minimum(0, d).sum())}}
