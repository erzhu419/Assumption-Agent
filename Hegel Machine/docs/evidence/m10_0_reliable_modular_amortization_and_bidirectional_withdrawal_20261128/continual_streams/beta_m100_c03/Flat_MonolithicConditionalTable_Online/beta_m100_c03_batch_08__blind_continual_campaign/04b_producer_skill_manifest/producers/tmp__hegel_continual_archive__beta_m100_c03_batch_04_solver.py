"""Box-simplex transport inversion with sealed walls and empirical rosette calibration."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kap, vel, h, rob):
    kh = (kap[:-1] + kap[1:]) * (0.5 / h**2)
    down = np.r_[-kh[:-1] - np.maximum(vel[1:-1], 0) / h, -kap[-1] / h**2]
    up = np.r_[-kap[0] / h**2, -kh[1:] - np.maximum(-vel[1:-1], 0) / h]
    main = np.r_[kap[0] / h**2 + rob / h, -(down[:-1] + up[1:]), kap[-1] / h**2 + rob / h]
    return diags((down, main, up), (-1, 0, 1), format='csc')


def _rosette_features(x, bank):
    a, b, c = x[:, bank], x[:, (bank+1)%4], x[:, (bank-1)%4]
    return np.column_stack((np.ones(len(a)), a, b, c, a*b, a**3))


def _calibrate(study, raw):
    sweeps = study['sensor_calibration_sweeps']
    x = np.asarray(sweeps['ref_rosette'])
    y = np.asarray(sweeps['raw_rosette'])
    sweeps = sweeps['single_bank_sweeps']
    matrices, readings = [], []
    for j in range(4):
        mask = sweeps['bank_id'] == j
        single = np.zeros((mask.sum(), 4))
        single[:, j] = sweeps['ref_signal'][mask]
        matrices.append(np.vstack((_rosette_features(x,j), _rosette_features(single,j))))
        readings.append(np.r_[y[:,j], sweeps['raw_readout'][mask]])
    def robust(A, b):
        keep = np.ones(len(b), dtype=bool)
        for _ in range(12):
            c = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
            e = np.abs(A@c-b)
            new = e <= max(6*np.median(e),1e-10)
            if np.array_equal(new, keep):
                break
            keep = new
        return np.linalg.lstsq(A[keep],b[keep],rcond=None)[0], keep
    fits = [robust(A,b) for A,b in zip(matrices,readings)]
    mat = np.asarray([c for c,_ in fits]); offset, gain, nxt, prv, cross, cubic = mat.T
    def predict(z):
        return offset + gain*z + nxt*np.roll(z,-1,axis=1) + prv*np.roll(z,1,axis=1) + cross*z*np.roll(z,-1,axis=1) + cubic*z**3
    yy = np.asarray(raw).reshape(-1,4)
    j = np.arange(4)
    jac0 = np.diag(gain)
    jac0[j,(j+1)%4] = nxt
    jac0[j,(j-1)%4] = prv
    z = np.linalg.solve(jac0,(yy-offset).T).T
    for _ in range(20):
        e = predict(z) - yy
        if np.max(np.abs(e)) < 1e-13:
            break
        jac = np.broadcast_to(jac0,(len(z),4,4)).copy()
        jac[:,j,j] += cross*np.roll(z,-1,axis=1) + 3*cubic*z**2
        jac[:,j,(j+1)%4] += cross*z
        z -= np.linalg.solve(jac,e[...,None])[...,0]
    return z.ravel()


def _profile(study, values):
    z = np.asarray(study['Z_nuisance'])
    return values - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    # The apparatus state preceding this campaign repeats the sealed-wall state.
    ax, ay = (_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                    study['L' + d] / (n - 1), 0.0)
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
