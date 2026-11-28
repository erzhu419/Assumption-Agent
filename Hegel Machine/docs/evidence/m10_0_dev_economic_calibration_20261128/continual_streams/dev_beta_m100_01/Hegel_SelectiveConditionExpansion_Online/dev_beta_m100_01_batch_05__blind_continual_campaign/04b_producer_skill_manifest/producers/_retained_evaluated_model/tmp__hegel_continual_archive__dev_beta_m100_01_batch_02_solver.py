"""Periodic transport and constrained source inversion in calibrated coordinates."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    i = np.arange(kappa.size)
    prev, nxt = (i - 1) % i.size, (i + 1) % i.size
    face = (kappa + kappa[nxt]) / (2 * h * h)
    left = face[prev] + np.maximum(velocity, 0) / h
    right = face + np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, prev, nxt])),
                             shape=(i.size, i.size))


def _forward(st):
    nx, ny = st['nx'], st['ny']
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx'] / (nx - 1))
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly'] / (ny - 1))
    L = (sparse.kron(sparse.eye(ny), ax, format='csc')
         + sparse.kron(ay, sparse.eye(nx), format='csc')
         + sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = st['sensor_ij']
    at = ij[:, 0] + nx * ij[:, 1]
    sampled = []
    for mu in st['mu_shifts']:
        U = splu(L + mu * eye).solve(rhs)
        sampled.append(U[at])
    return np.vstack(sampled), U.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(st, S):
    Z = np.asarray(st['Z_nuisance'])
    both = np.column_stack((S, st['y_obs_raw']))
    gram = Z.T @ Z + st['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = both - Z @ np.linalg.solve(gram, Z.T @ both)
    return projected[:, :-1], projected[:, -1]


def _qp(A, b, ridge, cap, budget):
    n = A.shape[1]
    H = A.T @ A + ridge * np.eye(n)
    c = A.T @ b

    def grad(w):
        return H @ w - c

    if budget == 0:
        w = np.zeros(n)
        return w, float(grad(w).min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(grad(w).max())

    # Scale the quadratic, then solve its exact equality system on the active face.
    scale = np.linalg.norm(H, ord=np.inf)
    res = minimize(lambda w: (0.5 * w @ H @ w - c @ w) / scale,
                   np.full(n, budget / n), jac=lambda w: grad(w) / scale,
                   method='SLSQP', bounds=[(0., cap)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                'jac': lambda w: np.ones(n)},
                   options={'ftol': 1e-14, 'maxiter': 500})
    w = res.x
    lower, upper = w <= 1e-8 * cap, w >= (1 - 1e-8) * cap
    free = ~(lower | upper)

    def gap(v, nu):
        d = grad(v) - nu
        return np.maximum(d, 0) @ v + np.maximum(-d, 0) @ (cap - v)

    g = grad(w)
    if np.any(free):
        nu = float(g[free].mean())
        v = np.where(upper, cap, 0.)
        hf = H[np.ix_(free, free)]
        rhs = c[free] - H[free] @ v
        x, direction = np.linalg.solve(hf, np.column_stack((rhs, np.ones(rhs.size)))).T
        mu = (budget - v.sum() - x.sum()) / direction.sum()
        v[free] = x + mu * direction
        if v.min() >= 0 and v.max() <= cap and gap(v, mu) <= gap(w, nu):
            return v, float(mu)
    else:
        nu = float((g[upper].max() + g[lower].min()) / 2)
        v = np.where(upper, cap, 0.)
        if v.sum() == budget and gap(v, nu) <= gap(w, nu):
            return v, nu
    return w, nu


def solve_downstream(cal_item: dict) -> dict:
    S, U_last = _forward(cal_item)
    A, b = _project(cal_item, S)
    ridge, cap, budget = (cal_item[key] for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(A, b, ridge, cap, budget)
    residual = A @ w - b
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = A.T @ residual + ridge * w
    d = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': objective,
                                'dual_lower_bound': float(objective - g @ w + nu * budget
                                                          + cap * np.minimum(0, d).sum())}}
