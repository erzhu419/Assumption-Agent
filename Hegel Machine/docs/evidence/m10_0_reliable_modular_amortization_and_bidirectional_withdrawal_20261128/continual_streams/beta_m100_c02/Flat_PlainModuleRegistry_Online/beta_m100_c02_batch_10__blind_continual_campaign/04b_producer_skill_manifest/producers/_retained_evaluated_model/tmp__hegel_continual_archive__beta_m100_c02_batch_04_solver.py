"""Conservative wrapped-x, closed-y transport and box-simplex inversion."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, wrapped=False):
    face = 0.5 * (kappa[:-1] + kappa[1:]) / h**2
    v = velocity[1:-1] / h
    west = -face[:-1] - np.maximum(v, 0)
    east = -face[1:] - np.maximum(-v, 0)
    if wrapped:
        near_left = face[0] + max(-velocity[0], 0) / h
        near_right = face[-1] + max(velocity[-1], 0) / h
        seam = 0.5 * (kappa[0] + kappa[-1]) / h**2
        wrap_left = seam + max(velocity[0], 0) / h
        wrap_right = seam + max(-velocity[-1], 0) / h
    else:
        near_left, near_right = kappa[[0, -1]] / h**2
        wrap_left = wrap_right = 0.0
    a = sparse.diags((np.r_[west, -near_right],
                      np.r_[near_left + wrap_left, -west - east, near_right + wrap_right],
                      np.r_[-near_left, east]), (-1, 0, 1), format='lil')
    if wrapped:
        a[0, -1] -= wrap_left
        a[-1, 0] -= wrap_right
    return a.tocsc()


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), wrapped=True)
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    op = (sparse.kron(sparse.eye(ny), ax, format='csc')
          + sparse.kron(ay, sparse.eye(nx), format='csc')
          + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    readings = []
    for mu in study['mu_shifts']:
        response = splu(op + mu * eye).solve(rhs)
        readings.append(response[i + nx * j])
    return np.concatenate(readings), response.T.reshape((-1, nx, ny), order='F')


def box_simplex(hess, linear, cap, budget):
    """Minimize 0.5*w@hess@w + linear@w, keeping iterates feasible."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hess @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    # -1 and +1 denote active lower and upper bounds.
    state = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(hess, np.inf) * cap,
                                         np.linalg.norm(linear, np.inf))
    for _ in range(20 * n * n):
        f = np.flatnonzero(state == 0)
        a = np.flatnonzero(state != 0)
        if f.size:
            hf = hess[np.ix_(f, f)]
            c = linear[f] + hess[np.ix_(f, a)] @ w[a]
            v, q = np.linalg.solve(hf, np.column_stack((-c, np.ones(f.size)))).T
            nu = (budget - w[a].sum() - v.sum()) / q.sum()
            target = v + nu * q
            if f.size == 1:
                target[0] = np.clip(budget - w[a].sum(), 0, cap)
            d = target - w[f]
            alpha = np.full(f.size, np.inf)
            low, high = (target < 0) & (d < 0), (target > cap) & (d > 0)
            alpha[low] = -w[f[low]] / d[low]
            alpha[high] = (cap - w[f[high]]) / d[high]
            k = int(alpha.argmin())
            if alpha[k] < 1:
                w[f] += alpha[k] * d
                state[f[k]] = -1 if low[k] else 1
                w[f[k]] = 0.0 if low[k] else cap
                continue
            w[f] = np.clip(target, 0, cap)
        g = hess @ w + linear
        if not f.size:
            lower = np.flatnonzero(state < 0)
            upper = np.flatnonzero(state > 0)
            l, u = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[u] - g[l] <= tol:
                return w, float((g[u] + g[l]) / 2)
            state[[l, u]] = 0
            continue
        violation = state * (g - nu)
        k = int(violation.argmax())
        if violation[k] <= tol:
            return w, float(nu)
        state[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Solve using the observations in canonical coordinates."""
    s = cal_item
    response, terminal = chamber_response(s)
    y = np.asarray(s['y_obs_raw'], dtype=float)
    z = np.asarray(s['Z_nuisance'], dtype=float)
    p = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    st = response - z @ (p @ response)
    ry = y - z @ (p @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    hess = st.T @ st + ridge * np.eye(st.shape[1])
    w, nu = box_simplex(hess, -(st.T @ ry), cap, budget)
    residual = st @ w - ry
    g = st.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    obj = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = obj - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': float(obj), 'dual_lower_bound': float(dual)}}
