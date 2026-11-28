"""Periodic transport and certified box-simplex inversion for this campaign."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _periodic_axis(kappa, velocity, h):
    # Entry i of each face array refers to the row i, including the seam.
    left = (kappa + np.roll(kappa, 1)) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + np.roll(kappa, -1)) / (2 * h**2) + np.maximum(-velocity, 0) / h
    i = np.arange(len(kappa))
    n = len(i)
    return csc_matrix((np.r_[left + right, -left, -right],
                       (np.tile(i, 3), np.r_[i, (i - 1) % n, (i + 1) % n])),
                      shape=(n, n))


def _responses(study):
    nx, ny = study['nx'], study['ny']
    ax = _periodic_axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _periodic_axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    identity = eye(nx * ny, format='csc')
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    modes = study['source_modes']
    rhs = modes.reshape(len(modes), nx * ny, order='F').T
    ix, iy = np.asarray(study['sensor_ij']).T
    observed = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        observed.append(fields[ix + nx * iy])
    return np.vstack(observed), fields.T.reshape(len(modes), nx, ny, order='F')


def _box_simplex(H, c, upper, total):
    n = len(c)
    if total == 0:
        return np.zeros(n), float(np.min(c))
    if total == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w + c))

    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)  # -1 lower bound, +1 upper bound
    tol = 128 * np.finfo(float).eps * max(1., np.max(np.abs(c)), upper * np.linalg.norm(H, np.inf))
    for _ in range(20 * n * n):
        g = H @ w + c
        free = np.flatnonzero(active == 0)
        if free.size:
            # Newton direction on the current face, constrained to sum to zero.
            f = len(free)
            kkt = np.empty((f + 1, f + 1))
            kkt[:f, :f] = H[np.ix_(free, free)]
            kkt[:f, f] = -1
            kkt[f, :f] = 1
            kkt[f, f] = 0
            step = np.linalg.solve(kkt, np.r_[-g[free], 0.])
            delta, nu = step[:-1], step[-1]
            proposed = w[free] + delta
            fraction = np.ones(f)
            low = (proposed < 0) & (delta < 0)
            high = (proposed > upper) & (delta > 0)
            fraction[low] = -w[free[low]] / delta[low]
            fraction[high] = (upper - w[free[high]]) / delta[high]
            hit = int(np.argmin(fraction))
            w[free] += fraction[hit] * delta
            if fraction[hit] < 1:
                j = free[hit]
                active[j] = -1 if low[hit] else 1
                w[j] = 0 if low[hit] else upper
                continue
            g = H @ w + c
        else:
            nu = (np.max(g[active > 0]) + np.min(g[active < 0])) / 2
        violation = active * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    S, terminal_modes = _responses(cal_item)
    y = np.asarray(cal_item['y_obs_raw']).ravel()
    Z = np.asarray(cal_item['Z_nuisance'])
    ridge = cal_item['source_ridge']
    upper, total = cal_item['u_box'], cal_item['total_budget']
    data = np.column_stack((S, y))
    coefficients = np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]),
                                   Z.T @ data)
    residualized = data - Z @ coefficients
    A, b = residualized[:, :-1], residualized[:, -1]
    H = A.T @ A + ridge * np.eye(A.shape[1])
    w, nu = _box_simplex(H, -(A.T @ b), upper, total)
    residual = A @ w - b
    g = A.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = g - nu
    dual = objective - g @ w + nu * total + upper * np.minimum(0, reduced).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': float(objective),
                                'dual_lower_bound': float(dual)}}
