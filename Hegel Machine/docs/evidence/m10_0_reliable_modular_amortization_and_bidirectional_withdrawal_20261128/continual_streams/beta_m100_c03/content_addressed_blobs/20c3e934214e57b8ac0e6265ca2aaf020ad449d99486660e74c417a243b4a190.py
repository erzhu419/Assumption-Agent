"""Transport–reaction inversion with periodic x and sealed y walls."""

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    face = (kappa + kappa[right]) / (2 * h**2)
    lo = face[left] + np.maximum(velocity, 0) / h
    hi = face + np.maximum(-velocity, 0) / h
    if not periodic:
        lo[0] = hi[-1] = 0.0
        hi[0], lo[-1] = kappa[0] / h**2, kappa[-1] / h**2
    return coo_matrix((np.r_[lo + hi, -lo, -hi],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n)).tocsc()


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                    study['L' + d] / (n - 1), periodic=(d == 'x')) for d, n in zip('xy', (nx, ny))]
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx * sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(indices) == len(study['y_obs_raw']) and len(shifts) > 1:
        indices = indices.reshape(len(shifts), -1)
    identity = eye(nx * ny, format='csc')
    samples = []
    for k, mu in enumerate(shifts):
        field = splu(operator + mu * identity).solve(rhs)
        samples.append(field[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(samples), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _box_qp(h, linear, cap, budget):
    """Feasible active-set Newton method for one equality and box bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = h @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    state = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1, np.max(np.abs(linear)), cap * np.linalg.norm(h, np.inf))
    for _ in range(20 * n + 100):
        f = np.flatnonzero(state == 0)
        g = h @ w + linear
        if not len(f):
            lower, upper = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[lo] >= g[hi] - tol:
                return w, float((g[lo] + g[hi]) / 2)
            state[[lo, hi]] = 0
            continue
        fixed = np.flatnonzero(state)
        hf = h[np.ix_(f, f)]
        gf = linear[f] + h[np.ix_(f, fixed)] @ w[fixed]
        v, q = np.linalg.solve(hf, np.column_stack((-gf, np.ones(len(f))))).T
        nu = (budget - w[fixed].sum() - v.sum()) / q.sum()
        target = v + nu * q
        step = target - w[f]
        ratios = np.full(len(f), np.inf)
        below, above = (target < 0) & (step < 0), (target > cap) & (step > 0)
        ratios[below] = -w[f][below] / step[below]
        ratios[above] = (cap - w[f][above]) / step[above]
        j = np.argmin(ratios)
        if ratios[j] < 1:
            w[f] += ratios[j] * step
            state[f[j]] = -1 if below[j] else 1
            w[f[j]] = 0 if below[j] else cap
            continue
        w[f] = np.clip(target, 0, cap)
        g = h @ w + linear
        violation = state * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s, last = _responses(cal_item)
    z = np.asarray(cal_item['Z_nuisance'])
    combined = np.column_stack((s, cal_item['y_obs_raw']))
    combined -= z @ np.linalg.solve(z.T @ z + cal_item['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ combined)
    s, y = combined[:, :-1], combined[:, -1]
    ridge, cap, budget = (float(cal_item[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_qp(s.T @ s + ridge * np.eye(s.shape[1]), -(s.T @ y), cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': obj,
                                'dual_lower_bound': float(obj - g @ w + nu * budget + cap * np.minimum(0, reduced).sum())}}
