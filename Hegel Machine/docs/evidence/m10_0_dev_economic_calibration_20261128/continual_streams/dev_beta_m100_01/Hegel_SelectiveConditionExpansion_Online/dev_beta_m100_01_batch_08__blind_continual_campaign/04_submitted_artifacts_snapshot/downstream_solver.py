"""Campaign transport and box-simplex source inversion in calibrated coordinates."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    k = np.asarray(kappa)
    v = np.asarray(velocity)
    n = k.size
    i = np.arange(n)
    prev, nxt = (i - 1) % n, (i + 1) % n
    face = (k + k[nxt]) / (2 * h * h)
    left = face[prev] + np.maximum(v, 0) / h
    right = face + np.maximum(-v, 0) / h
    if not periodic:
        left[0] = right[-1] = 0
        right[0], left[-1] = k[0] / h**2, k[-1] / h**2
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, prev, nxt])), shape=(n, n))


def _forward(st):
    nx, ny = st['nx'], st['ny']
    # The campaign's calibrated tracer pulses identify a periodic x seam and
    # reflecting y walls with endpoint diffusivity and no Robin loss.
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx'] / (nx - 1), True)
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly'] / (ny - 1), False)
    L = (sparse.kron(sparse.eye(ny), ax, format='csc')
         + sparse.kron(ay, sparse.eye(nx), format='csc')
         + sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = np.asarray(st['sensor_ij'])
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
    if budget <= 0:
        return np.zeros(n), float((-c).min())
    if budget >= n * cap:
        w = np.full(n, cap)
        return w, float((H @ w - c).max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    tol = 64 * np.finfo(float).eps * max(1., np.linalg.norm(c, np.inf), cap * np.linalg.norm(H, np.inf))
    for _ in range(20 * n * n):
        f = np.flatnonzero(state == 0)
        lo = np.flatnonzero(state == -1)
        hi = np.flatnonzero(state == 1)
        if not f.size:
            g = H @ w - c
            l, u = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[u] <= g[l] + tol:
                return w, float((g[u] + g[l]) / 2)
            state[[l, u]] = 0
            continue
        # Equality-constrained minimizer on the current face.
        Hf = H[np.ix_(f, f)]
        rhs = c[f] - H[np.ix_(f, hi)].sum(axis=1) * cap
        x, q = np.linalg.solve(Hf, np.column_stack((rhs, np.ones(f.size)))).T
        nu = (budget - cap * hi.size - x.sum()) / q.sum()
        target = x + nu * q
        direction = target - w[f]
        ratios = np.full(f.size, np.inf)
        down = (target < 0) & (direction < 0)
        up = (target > cap) & (direction > 0)
        ratios[down] = -w[f[down]] / direction[down]
        ratios[up] = (cap - w[f[up]]) / direction[up]
        j = np.argmin(ratios)
        if ratios[j] < 1:
            w[f] += max(0., ratios[j]) * direction
            state[f[j]] = 1 if up[j] else -1
            w[f[j]] = cap if up[j] else 0.
            continue
        w[f] = np.clip(target, 0., cap)
        g = H @ w - c
        violation = state * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    st = cal_item
    S, U_last = _forward(st)
    A, b = _project(st, S)
    cap, budget, ridge = st['u_box'], st['total_budget'], st['source_ridge']
    w, nu = _qp(A, b, ridge, cap, budget)
    residual = A @ w - b
    g = A.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = np.maximum(0., g - nu)
    upper = np.maximum(0., nu - g)
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': obj,
                                'dual_lower_bound': float(obj - g @ w + nu * budget - cap * upper.sum())}}
