"""Canonical transport solve with a periodic x axis and a sealed y axis."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, *, periodic):
    """Row-local upwind interior, with periodic or sealed endpoint rows."""
    row = np.arange(len(kappa))
    prev = (row - 1) % len(row)
    nxt = (row + 1) % len(row)
    left = (kappa + kappa[prev]) / (2 * h**2) + np.maximum(velocity, 0) / h
    right = (kappa + kappa[nxt]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    if not periodic:
        left[0], right[-1] = 0., 0.
        right[0], left[-1] = kappa[0] / h**2, kappa[-1] / h**2
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(row, 3), np.r_[row, prev, nxt])),
                             shape=(len(row), len(row)))


def _forward(st):
    nx, ny = st['nx'], st['ny']
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx'] / (nx - 1), periodic=True)
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly'] / (ny - 1), periodic=False)
    L = (sparse.kron(sparse.eye(ny), ax, format='csc') +
         sparse.kron(ay, sparse.eye(nx), format='csc') +
         sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = st['sensor_ij'].T
    at = ix + nx * iy
    samples = []
    for mu in st['mu_shifts']:
        u = splu(L + mu * identity).solve(rhs)
        samples.append(u[at])
    return np.vstack(samples), u.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(st, S, y):
    Z = st['Z_nuisance']
    data = np.column_stack((S, y))
    coefficients = np.linalg.solve(Z.T @ Z + st['nuisance_ridge'] * np.eye(Z.shape[1]),
                                   Z.T @ data)
    projected = data - Z @ coefficients
    return projected[:, :-1], projected[:, -1]


def _qp(A, b, ridge, cap, budget):
    """Feasible active-set solve; at each face solve the equality QP exactly."""
    n = A.shape[1]
    H = A.T @ A + ridge * np.eye(n)
    c = A.T @ b
    if budget == 0:
        return np.zeros(n), float(np.min(-c))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w - c))

    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1 at zero, +1 at cap
    tol = 64 * np.finfo(float).eps * max(1., cap * np.linalg.norm(H, np.inf),
                                        np.linalg.norm(c, np.inf))
    for _ in range(20 * n * n):
        free = active == 0
        if not free.any():
            g = H @ w - c
            low = np.argmin(np.where(active < 0, g, np.inf))
            high = np.argmax(np.where(active > 0, g, -np.inf))
            if g[high] <= g[low] + tol:
                return w, float((g[low] + g[high]) / 2)
            active[[low, high]] = 0
            continue

        fixed = ~free
        linear = c[free] - H[np.ix_(free, fixed)] @ w[fixed]
        x, p = np.linalg.solve(H[np.ix_(free, free)],
                               np.column_stack((linear, np.ones_like(linear)))).T
        nu = (budget - w[fixed].sum() - x.sum()) / p.sum()
        target = x + nu * p
        wt = w[free]
        direction = target - wt
        hit = np.ones_like(target)
        low, high = target < 0, target > cap
        hit[low] = -wt[low] / direction[low]
        hit[high] = (cap - wt[high]) / direction[high]
        if np.any(low | high):
            j = np.argmin(np.where(low | high, hit, np.inf))
            w[free] = wt + hit[j] * direction
            k = np.flatnonzero(free)[j]
            active[k] = -1 if low[j] else 1
            w[k] = 0. if low[j] else cap
            continue

        w[free] = target
        g = H @ w - c
        violation = active * (g - nu)
        k = np.argmax(violation)
        if violation[k] <= tol:
            return w, float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Solve in canonical sensor coordinates and certify the box-simplex optimum."""
    S, U_last = _forward(cal_item)
    A, b = _project(cal_item, S, cal_item['y_obs_raw'])
    ridge = cal_item['source_ridge']
    cap = cal_item['u_box']
    budget = cal_item['total_budget']
    w, nu = _qp(A, b, ridge, cap, budget)
    residual = A @ w - b
    gradient = A.T @ residual + ridge * w
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = gradient - nu
    dual = float(primal - gradient @ w + nu * budget + cap * np.minimum(0., reduced).sum())
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': float(nu),
            'lam_lower': np.maximum(0., reduced).tolist(),
            'lam_upper': np.maximum(0., -reduced).tolist(),
            'primal_objective': primal,
            'dual_lower_bound': dual,
        },
    }
