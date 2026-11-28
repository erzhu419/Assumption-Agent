"""Predict scalar responses and trajectories from calibrated panel interactions."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    bases = [_affine(np.column_stack((x, z, x*z, x**3)))
             for x, z in zip(u.T, np.roll(u, -1, axis=1).T)]
    ids = np.arange(len(u))
    for _ in range(8):
        coef = np.column_stack([_fit(f[ids], v[ids, j]) for j, f in enumerate(bases)])
        err = np.linalg.norm(np.column_stack([f @ coef[:, j] for j, f in enumerate(bases)])-v, axis=1)
        new_ids = np.flatnonzero(err < max(.002, 4*np.median(err)))
        if np.array_equal(ids, new_ids):
            break
        ids = new_ids
    return coef


def _invert(x, coef):
    c0, c1, c2, c3, c4 = coef
    def residual(z, observed):
        zn = np.roll(z, -1)
        return c0 + c1*z + c2*zn + c3*z*zn + c4*z**3 - observed
    def jac(z, observed):
        d = np.diag(c1 + c3*np.roll(z, -1) + 3*c4*z*z)
        d[np.arange(4), np.roll(np.arange(4), -1)] = c2+c3*z
        return d
    return np.array([least_squares(residual, (row-c0)/c1, jac=jac, args=(row,),
                                   gtol=1e-12, ftol=1e-12, xtol=1e-12).x for row in x])


def _views(study, z):
    pb, pc = study['panel_b'], study['panel_c']
    b = _affine(z) @ _fit(_affine(pb['u']), pb['v'])
    u, v = pc['u'], pc['v1']
    def soft(x, k):
        w = k[0]*x + k[1]
        return _affine(w/(1+np.abs(w)) + k[2]*x)
    def res(k):
        f = soft(u, k)
        return (f @ _fit(f, v)-v).ravel()
    k = least_squares(res, [1.5, 0., 0.], bounds=([.15, -4, -4], [7., 4, 4]), max_nfev=100).x
    p = soft(z, k) @ _fit(soft(u, k), v)
    def cosine(x, freq):
        return _affine(np.cos(freq*x))
    def loss(freq):
        f = cosine(u, freq)
        return np.mean((f @ _fit(f, pc['v2']) - pc['v2'])**2)
    freq = minimize_scalar(loss, bounds=(2., 2.8), method='bounded').x
    q = cosine(z, freq) @ _fit(cosine(u, freq), pc['v2'])
    return b, p, q


def _outer(u, v):
    return (u[:, :, None]*v[:, None, :]).reshape(len(u), -1)


def _design(study, x):
    z = _invert(x, _calibrate(study['panel_a']))
    b, p, q = _views(study, z)
    return np.column_stack((z, p, q, _outer(b, p), _outer(p, q)))


def _predict(design, values, train, test, penalty):
    a = design[train]
    center = a.mean(axis=0)
    scale = a.std(axis=0)
    f = (a-center)/scale
    offset = values[train].mean(axis=0)
    gram = f.T @ f + penalty*np.eye(f.shape[1])
    w = np.linalg.solve(gram, f.T @ (values[train]-offset))
    return (design[test]-center)/scale @ w + offset


def solve_batch(study: dict) -> dict:
    data = study['train']
    n = len(data['x'])
    design = _design(study, np.vstack((data['x'], study['test']['x'])))
    y = _predict(design, data['y'], slice(n), slice(n, None), .003)
    state = _predict(design, data['traj'], slice(n), slice(n, None), .0001)
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
