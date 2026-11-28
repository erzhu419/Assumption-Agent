"""Calibrated chamber transport and box-simplex source reconstruction."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    """Row-local upwind transport with a periodic seam."""
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    n = len(kappa)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face - np.maximum(vel[1:], 0) / h
    upper = -face - np.maximum(-vel[:-1], 0) / h
    seam = (kappa[-1] + kappa[0]) / (2 * h**2)
    wrap_first = seam + max(vel[0], 0) / h
    wrap_last = seam + max(-vel[-1], 0) / h
    diag = np.r_[-upper[0] + wrap_first, -lower[:-1] - upper[1:],
                 -lower[-1] + wrap_last]
    a = sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')
    a += sparse.csc_matrix(([-wrap_first, -wrap_last], ([0, n-1], [n-1, 0])),
                           shape=(n, n))
    return a


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    ix, iy = sparse.eye(nx, format='csc'), sparse.eye(ny, format='csc')
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = sparse.kron(iy, ax, format='csc') + sparse.kron(ay, ix, format='csc')
    rhs = np.column_stack([np.asarray(mode).ravel(order='F') for mode in study['source_modes']])
    ij = np.asarray(study['sensor_ij'])
    rows = ij[:, 0] + nx * ij[:, 1]
    sampled = []
    for shift in study['mu_shifts']:
        u = splu(operator + sparse.diags(reaction + shift, format='csc')).solve(rhs)
        sampled.append(u[rows])
    return np.vstack(sampled), u.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(study, S, y):
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    values = np.column_stack((S, y))
    values -= z @ np.linalg.solve(gram, z.T @ values)
    return values[:, :-1], values[:, -1]


def _box_simplex(H, f, cap, budget):
    """Primal active-set method with exact minimization on each free face."""
    n = len(f)
    if budget <= 0 or budget >= n * cap:
        w = np.full(n, 0.0 if budget <= 0 else cap)
        grad = H @ w + f
        return w, float(grad.min() if budget <= 0 else grad.max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)
    tol = 32 * np.finfo(float).eps * max(1.0, cap)
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        fixed = np.flatnonzero(state != 0)
        if len(free):
            A = H[np.ix_(free, free)]
            rhs = -f[free] - H[np.ix_(free, fixed)] @ w[fixed]
            cf = np.linalg.solve(A, np.column_stack((rhs, np.ones(len(free)))))
            nu = (budget - w[fixed].sum() - cf[:, 0].sum()) / cf[:, 1].sum()
            dest = cf[:, 0] + nu * cf[:, 1]
            delta = dest - w[free]
            hits = np.full(len(free), np.inf)
            down = dest < -tol
            up = dest > cap + tol
            hits[down] = -w[free[down]] / delta[down]
            hits[up] = (cap - w[free[up]]) / delta[up]
            j = int(np.argmin(hits))
            if hits[j] < 1.0:
                w[free] += max(0.0, hits[j]) * delta
                state[free[j]] = -1 if down[j] else 1
                w[free[j]] = 0.0 if down[j] else cap
                continue
            w[free] = dest
        grad = H @ w + f
        if not len(free):
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            a, b = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[a] >= grad[b] - tol:
                return w, float((grad[a] + grad[b]) / 2)
            state[[a, b]] = 0
            continue
        violation = state * (grad - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol * max(1.0, np.max(np.abs(grad))):
            return np.clip(w, 0.0, cap), float(nu)
        state[j] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Solve in the coordinates supplied by the verified upstream transducer."""
    s, u_last = _forward(cal_item)
    s, y = _project(cal_item, s, np.asarray(cal_item['y_obs_raw']).ravel())
    cap, budget, ridge = (float(cal_item[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    res = s @ w - y
    g = s.T @ res + ridge * w
    obj = float((res @ res + ridge * (w @ w)) / 2)
    delta = g - nu
    dual = float(obj - g @ w + nu * budget + cap * np.minimum(0., delta).sum())
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, u_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0., delta).tolist(),
            'lam_upper': np.maximum(0., -delta).tolist(),
            'primal_objective': obj, 'dual_lower_bound': dual,
        },
    }
