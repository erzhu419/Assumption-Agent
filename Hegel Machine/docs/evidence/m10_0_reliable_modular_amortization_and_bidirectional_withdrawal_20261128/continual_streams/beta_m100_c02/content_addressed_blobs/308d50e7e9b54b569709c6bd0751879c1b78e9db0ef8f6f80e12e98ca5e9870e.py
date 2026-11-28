"""Periodic transport-reaction inversion in calibrated sensor coordinates."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    i = np.arange(len(kappa))
    west = (i - 1) % len(i)
    east = (i + 1) % len(i)
    a = (kappa[i] + kappa[west]) / (2*h*h) + np.maximum(velocity, 0) / h
    b = (kappa[i] + kappa[east]) / (2*h*h) + np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[a+b, -a, -b],
                             (np.tile(i, 3), np.r_[i, west, east])),
                             shape=(len(i), len(i)))


def _responses(s):
    nx, ny = s['nx'], s['ny']
    ax = _axis(s['kappa_x'], s['vel_x'], s['Lx'] / (nx-1))
    ay = _axis(s['kappa_y'], s['vel_y'], s['Ly'] / (ny-1))
    transport = sparse.kronsum(ax, ay, format='csc')
    reaction = s['V_reaction_2d'].ravel(order='F')
    modes = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
    i, j = s['sensor_ij'].T
    observations = []
    for mu in s['mu_shifts']:
        matrix = transport + sparse.diags(reaction + mu, format='csc')
        u = splu(matrix).solve(modes)
        observations.append(u[i + nx*j])
    return np.vstack(observations), u.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(h, c, cap, budget):
    """Minimize 0.5*w.T@h@w + c.T@w by feasible active-set steps."""
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(np.min(c))
    if budget == n*cap:
        w = np.full(n, cap)
        return w, float(np.max(h @ w + c))
    w = np.full(n, budget/n)
    status = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    tol = 32 * np.finfo(float).eps * max(1., np.max(np.abs(c)), cap*np.linalg.norm(h, np.inf))
    for _ in range(20*n*n):
        free = np.flatnonzero(status == 0)
        lo = np.flatnonzero(status < 0)
        hi = np.flatnonzero(status > 0)
        if free.size == 0:
            g = h @ w + c
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[i]+g[j])/2)
            status[[i, j]] = 0
            continue
        m = free.size
        system = np.empty((m+1, m+1))
        system[:m, :m] = h[np.ix_(free, free)]
        system[:m, m] = -1
        system[m, :m] = 1
        system[m, m] = 0
        optimum = np.linalg.solve(system, np.r_[-c[free] - h[np.ix_(free, hi)] @ w[hi], budget-cap*len(hi)])
        target, nu = optimum[:-1], optimum[-1]
        current = w[free]
        below = (target < 0) & (target < current)
        above = (target > cap) & (target > current)
        alpha = np.ones(m)
        alpha[below] = current[below] / (current[below]-target[below])
        alpha[above] = (cap-current[above]) / (target[above]-current[above])
        k = int(np.argmin(alpha))
        if alpha[k] < 1 - 8*np.finfo(float).eps:
            w[free] = np.clip(current + alpha[k]*(target-current), 0, cap)
            status[free[k]] = 1 if above[k] else -1
            w[free[k]] = cap if above[k] else 0
            continue
        w[free] = np.clip(target, 0, cap)
        violations = status * (h @ w + c - nu)
        k = int(np.argmax(violations))
        if violations[k] <= tol:
            return w, float(nu)
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    response, fields = _responses(s)
    z = s['Z_nuisance']
    y = s['y_obs_raw']
    m = response.shape[1]
    data = np.column_stack((response, y))
    projected = data - z @ np.linalg.solve(
        z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    st, ry = projected[:, :m], projected[:, m]
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    h = st.T @ st + ridge * np.eye(m)
    w, nu = _box_simplex(h, -(st.T @ ry), cap, budget)
    r = st @ w - ry
    g = st.T @ r + ridge*w
    objective = 0.5*(r @ r + ridge*(w @ w))
    delta = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0, delta).tolist(),
            'lam_upper': np.maximum(0, -delta).tolist(),
            'primal_objective': float(objective),
            'dual_lower_bound': float(objective - g @ w + nu*budget + cap*np.minimum(0, delta).sum()),
        },
    }
