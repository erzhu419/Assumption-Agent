"""Recover panel operators and fit the scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import root, least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _cal_basis(z):
    return np.stack((np.ones_like(z), z, np.tanh(1.8*z),
                     np.sin(2.4*np.roll(z, 2, axis=-1))), axis=-1)


def _coordinates(study, x):
    a, b, c = (study['panel_' + k] for k in 'abc')
    d, v = _cal_basis(a['u']), a['v']
    keep = np.arange(len(v))
    for _ in range(10):
        coef = np.array([_fit(d[keep, j], v[keep, j]) for j in range(4)])
        residual = np.linalg.norm((d*coef).sum(-1)-v, axis=1)
        keep = np.argsort(residual)[:max(4, int(.85*len(v)))]
    keep = residual <= max(.001, 8*np.median(residual))
    coef = np.array([_fit(d[keep, j], v[keep, j]) for j in range(4)])

    def invert(row):
        def fun(z):
            return (_cal_basis(z)*coef).sum(-1)-row
        r = root(fun, row, tol=1e-10)
        if np.max(np.abs(r.fun)) > 1e-8:
            r = least_squares(fun, row, gtol=1e-12)
        return r.x

    def dead(u):
        return _aff(np.sign(u)*np.maximum(np.abs(u)-.35, 0)**1.5)

    def pairs(u):
        return _aff(np.column_stack([u[:, i]*u[:, j]
                                      for i, j in ((0,1),(0,3),(1,2),(2,3))]))

    z = np.array([invert(row) for row in x])
    w = dead(z) @ _fit(dead(b['u']), b['v'])
    return pairs(w) @ _fit(pairs(c['u']), c['v1'])


def _scalar_basis(q):
    # Retain only the two opposite-coordinate products supported by CV.
    return _aff(np.column_stack((q, q[:, 0]*q[:, 2], q[:, 1]*q[:, 3])))


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    q = _coordinates(study, np.concatenate((tr['x'], study['test']['x'])))
    d = _scalar_basis(q)
    y = d[n:] @ _fit(d[:n], tr['y'])
    d = _aff(q)
    coeff = _fit(d[:n], tr['traj'])
    modes = np.linalg.svd(d[:n] @ coeff, full_matrices=False)[2][:2]
    state = (d[n:] @ coeff @ modes.T) @ modes
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
