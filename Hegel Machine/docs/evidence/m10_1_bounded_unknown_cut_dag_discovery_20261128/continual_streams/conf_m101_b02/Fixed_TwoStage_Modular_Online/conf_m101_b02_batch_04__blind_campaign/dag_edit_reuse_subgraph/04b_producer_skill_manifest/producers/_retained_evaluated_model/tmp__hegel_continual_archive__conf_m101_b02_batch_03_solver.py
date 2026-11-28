"""Recover latent coordinates and fit the two response readouts."""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar, root
from scipy.spatial.distance import cdist


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _calibration(x):
    i, j = np.triu_indices(x.shape[1], 1)
    return _design(x, x*x, x*x*x, x[:, i]*x[:, j])


def _coordinates(study, x):
    pa = study['panel_a']
    f, v = _calibration(pa['u']), pa['v']
    w = _ls(f, v)
    for _ in range(60):
        q = np.maximum(np.linalg.norm(v-f@w, axis=1), 0.00025)**-0.5
        w = _ls(f*q[:, None], v*q[:, None])
    inside = np.linalg.norm(v-f@w, axis=1) < 0.005
    w = _ls(f[inside], v[inside])
    x = np.array([root(lambda a: (_calibration(a[None])@w-y).ravel(), y).x for y in x])
    pb = study['panel_b']
    z = _design(x)@_ls(_design(pb['u']), pb['v'])
    pc = study['panel_c']
    ops = []
    def odd(u, k):
        return -np.sign(u)*np.expm1(-k*abs(u))
    def even(u, k):
        u2 = u*u
        return u2/np.sqrt(1+k*u2)
    for transform, name, bounds in [(odd, 'v1', (.5, 3)), (even, 'v2', (.05, 1.5))]:
        u, v = pc['u'], pc[name]
        def residual(rate):
            f = _design(u, transform(u, rate)) if name == 'v2' else _design(transform(u, rate))
            return np.linalg.norm(v-f@_ls(f, v))
        k = minimize_scalar(residual, bounds=bounds, method='bounded').x
        if name == 'v1':
            ops.append(_design(transform(z, k))@_ls(_design(transform(u, k)), v))
        else:
            ops.append(_design(z, transform(z, k))@_ls(_design(u, transform(u, k)), v))
    return np.column_stack((z, *ops))


def _kernel(x, y):
    i, j = np.triu_indices(x.shape[1])
    px, py = x[:, i]*x[:, j], y[:, i]*y[:, j]
    return (1 + x@y.T + .1*(px@py.T)
            + .4*np.exp(-cdist(x, y, metric='sqeuclidean')/60))


def _predict(v, n, y, state):
    # The second auxiliary output is useful, but its quadratic products
    # extrapolate less reliably than those of the primary and odd branches.
    weights = np.repeat([2., 2., .35], 4)
    z = v * (weights / np.maximum(v[:n].std(0), .001))
    train = z[:n]
    alpha = solve(_kernel(train, train) + .004*np.eye(n),
                  np.column_stack((y, state)), assume_a='pos')
    pred = _kernel(z[n:], train) @ alpha
    return pred[:, 0], pred[:, 1:]


def solve_batch(study: dict) -> dict:
    tr = study['train']
    v = _coordinates(study, np.vstack((tr['x'], study['test']['x'])))
    y, state = _predict(v, len(tr['x']), tr['y'], tr['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
