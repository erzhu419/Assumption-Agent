"""Source inversion in the calibrated periodic-x, sealed-y chamber."""

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
        lo[0], hi[0] = 0, kappa[0] / h**2
        lo[-1], hi[-1] = kappa[-1] / h**2, 0
    return coo_matrix((np.r_[lo + hi, -lo, -hi],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n)).tocsc()


def _forward(s):
    nx, ny = s['nx'], s['ny']
    ax, ay = (_axis(s['kappa_' + key], s['vel_' + key], s['L' + key] / (n - 1), key == 'x')
              for key, n in zip('xy', (nx, ny)))
    L = (kron(eye(ny), ax) + kron(ay, eye(nx))
         + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = s['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = s['sensor_ij']
    positions = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for mu in s['mu_shifts']:
        u = splu(L + mu * identity).solve(rhs)
        blocks.append(u[positions])
    return np.vstack(blocks), u.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, total):
    n = len(c)
    if total == 0 or total == n * cap:
        w = np.full(n, total / n)
        g = H @ w + c
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    for _ in range(20 * n + 20):
        free = state == 0
        g = H @ w + c
        if free.any():
            block = H[np.ix_(free, free)]
            ones = np.ones(free.sum())
            q, r = np.linalg.solve(block, np.column_stack((g[free], ones))).T
            nu = (q.sum() + total - w.sum()) / r.sum()
            p = nu * r - q
            wf = w[free]
            trial = wf + p
            hit = np.flatnonzero((trial < 0) | (trial > cap))
            if hit.size:
                targets = np.where(p[hit] < 0, 0, cap)
                steps = (targets - wf[hit]) / p[hit]
                k = np.argmin(steps)
                w[free] = np.clip(wf + steps[k] * p, 0, cap)
                j = np.flatnonzero(free)[hit[k]]
                w[j] = targets[k]
                state[j] = -1 if targets[k] == 0 else 1
                continue
            w[free] = trial
            g = H @ w + c
        else:
            nu = (g[state == -1].min() + g[state == 1].max()) / 2
        violation = state * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= 1e-12:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    S, U = _forward(s)
    y = s['y_obs_raw']
    Z = s['Z_nuisance']
    P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (P @ S)
    y = y - Z @ (P @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    reduced = g - nu
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    certificate = dict(weights=w.tolist(), nu_sum=nu,
                       lam_lower=np.maximum(reduced, 0).tolist(),
                       lam_upper=np.maximum(-reduced, 0).tolist(),
                       primal_objective=J,
                       dual_lower_bound=float(J - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()))
    return dict(weights=w, terminal_field=np.tensordot(w, U, axes=(0, 0)), kkt_certificate=certificate)
