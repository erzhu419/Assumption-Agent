"""Box-simplex transport inversion with sealed walls and empirical rosette calibration."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kap, vel, h):
    kh = (kap[:-1] + kap[1:]) * (0.5 / h**2)
    down = np.r_[-kh[:-1] - np.maximum(vel[1:-1], 0) / h, -kap[-1] / h**2]
    up = np.r_[-kap[0] / h**2, -kh[1:] - np.maximum(-vel[1:-1], 0) / h]
    main = np.r_[kap[0] / h**2, -(down[:-1] + up[1:]), kap[-1] / h**2]
    return diags((down, main, up), (-1, 0, 1), format='csc')


def _sensor_fit(study, keep=None):
    c = study['sensor_calibration_sweeps']
    x, y = np.asarray(c['ref_rosette']), np.asarray(c['raw_rosette'])
    clean = np.max(np.abs(y), axis=1) < 5
    if keep is not None:
        clean &= keep
    x, y = x[clean], y[clean]
    q = c['single_bank_sweeps']
    bank, sx, sy = (np.asarray(q[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    good = abs(sy) < 5
    pairs = [(bank == j) & good for j in range(4)]
    def features(a, b, f, h):
        return np.column_stack((np.ones_like(a), a, np.sin(f*a), b, 1-np.cos(h*b)))
    def fit(freq):
        coefficients, errors = [], []
        for j, m in enumerate(pairs):
            ax = np.r_[x[:, j], sx[m]]
            bx = np.r_[x[:, (j+2)%4], np.zeros(m.sum())]
            target = np.r_[y[:, j], sy[m]]
            d = features(ax, bx, *freq)
            p = np.linalg.lstsq(d, target, rcond=None)[0]
            coefficients.append(p)
            errors.append(d @ p - target)
        return np.array(coefficients), np.concatenate(errors)
    ff = least_squares(lambda f: fit(f)[1], [2.5, 2.], bounds=([1., .5], [4., 4.]), gtol=1e-13).x
    p, _ = fit(ff)
    def forward(x):
        return np.column_stack([features(x[:, j], x[:, (j+2)%4], *ff) @ p[j] for j in range(4)])
    return ff, p, forward


def _calibrate(study, raw):
    (f, h), p, forward = _sensor_fit(study)
    y = np.asarray(raw).reshape(-1, 4)
    x = (y - p[:, 0]) / (p[:, 1] + f*p[:, 2])
    for _ in range(30):
        r = forward(x) - y
        if abs(r).max() < 1e-12:
            break
        jac = np.zeros((len(x), 4, 4))
        j = np.arange(4)
        jac[:, j, j] = p[:, 1] + f*p[:, 2]*np.cos(f*x)
        other = (j+2)%4
        jac[:, j, other] = p[:, 3] + h*p[:, 4]*np.sin(h*x[:, other])
        x -= np.linalg.solve(jac, r[..., None])[..., 0]
    return x.ravel()


def _profile(study, values):
    z = np.asarray(study['Z_nuisance'])
    return values - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                    study['L' + d] / (n - 1))
              for d, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T.copy()
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx * sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(indices) == len(study['y_obs_raw']) and len(shifts) > 1:
        indices = indices.reshape(len(shifts), -1)
    sampled = []
    identity = eye(nx * ny, format='csc')
    for k, shift in enumerate(shifts):
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(sampled), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_qp(hessian, linear, cap, budget):
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, 0.0 if budget == 0 else cap)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    eps = 64 * np.finfo(float).eps
    grad_tol = eps * max(np.abs(linear).max(), cap * np.linalg.norm(hessian, ord=np.inf), 1e-100)
    for _ in range(100 + 10 * n):
        free = active == 0
        g = hessian @ w + linear
        if not free.any():
            lower = np.where(active < 0)[0]
            upper = np.where(active > 0)[0]
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + grad_tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue
        gf = g[free]
        hf = hessian[np.ix_(free, free)]
        q, v = np.linalg.solve(hf, np.column_stack((gf, np.ones(len(gf))))).T
        nu = (q.sum() + budget - w.sum()) / v.sum()
        step = nu * v - q
        wf = w[free]
        low = step < 0
        bound = np.where(low, 0., cap)
        travel = np.full(len(step), np.inf)
        np.divide(bound - wf, step, out=travel, where=np.abs(step) > eps * cap)
        hit = travel.argmin()
        if travel[hit] < 1 - eps:
            w[free] += max(0., travel[hit]) * step
            i = np.flatnonzero(free)[hit]
            w[i] = bound[hit]
            active[i] = -1 if low[hit] else 1
            continue
        w[free] = np.clip(wf + step, 0., cap)
        g = hessian @ w + linear
        nu = float(g[free].mean())
        violation = active * (g - nu)
        hit = violation.argmax()
        if violation[hit] <= grad_tol:
            return w, nu
        active[hit] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s, terminal = _responses(study)
    y = _calibrate(study, study['y_obs_raw'])
    projected = _profile(study, np.column_stack((s, y)))
    s, y = projected[:, :-1], projected[:, -1]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_qp(s.T @ s + ridge * np.eye(s.shape[1]), -(s.T @ y), cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    delta = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., delta).tolist(),
                                'lam_upper': np.maximum(0., -delta).tolist(),
                                'primal_objective': obj,
                                'dual_lower_bound': float(obj - g @ w + nu * budget + cap * np.minimum(0., delta).sum())}}
