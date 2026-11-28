from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _quadratic(x):
    return np.column_stack([x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)])


def _calibrate(panel, z):
    u, v1, v2 = (np.asarray(panel[k]) for k in ('u', 'v1', 'v2'))
    a = _design(_quadratic(u))
    p = np.linalg.lstsq(a, v1, rcond=None)[0]
    c = _design(_quadratic(z)) @ p
    def profile(k):
        x = _design(u, np.sqrt(1 + k*u*u))
        pinv = np.linalg.pinv(x)
        fit = pinv @ v2
        resid = x @ fit - v2
        leverage = np.einsum('ij,ji->i', x, pinv)
        return np.mean((resid.reshape(len(u), -1) / (1-leverage[:, None]))**2), fit, resid
    k = minimize_scalar(lambda k: profile(k)[0], bounds=(.1, 3.), method='bounded').x
    _, q, _ = profile(k)
    d = _design(z, np.sqrt(1 + k*z*z)) @ q
    return c, d


def _predict(features, target, length, nugget):
    n = len(target)
    z = (features-features[:n].mean(0))/(.1+features[:n].std(0))
    k = np.exp(-cdist(z, z[:n], 'sqeuclidean')/(2*length**2*z.shape[1]))
    k += 1 + (z@z[:n].T)/z.shape[1]
    return k[n:] @ solve(k[:n]+nugget*np.eye(n), target, assume_a='pos')


def run_node(inputs: dict) -> dict:
    s, upstream = inputs['__input__'], inputs['node_00']
    z, b = upstream['z'], upstream['b']
    c, d = _calibrate(s['panel_c'], z)
    y = _predict(np.c_[z, b, c, d, _quadratic(z), _quadratic(b)], s['train']['y'], 3.5, 1e-5)
    traj = _predict(np.c_[z, b], s['train']['traj'], 2., 1e-5)
    diagnostics = np.array([y.mean(), y.std(), traj[:, -1].mean(), traj[:, -1].std()])
    return {'y_hat': y, 'state_pred': traj, 'diagnostics_vec': diagnostics}
