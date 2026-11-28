"""Panel-calibrated spatial prediction with a shared trajectory response."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def _coordinates(study, x):
    a = study['panel_a']
    u, v = np.asarray(a['u']), np.asarray(a['v'])
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slope = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)
    offset = np.median(v - u*slope, axis=0)
    for _ in range(3):
        r = np.linalg.norm(v - u*slope - offset, axis=1)
        keep = r <= max(6*np.median(r), 0.0005)
        if keep.sum() < 5:
            break
        for k in range(u.shape[1]):
            slope[k], offset[k] = np.linalg.lstsq(np.c_[u[keep,k], np.ones(keep.sum())], v[keep,k], rcond=None)[0]
    b = study['panel_b']
    bu, bv = np.asarray(b['u']), np.asarray(b['v'])
    bmap = np.linalg.lstsq(np.c_[np.ones(len(bu)), bu], bv, rcond=None)[0][1:]
    safe = np.where(abs(slope)>1e-8, slope, 1.)
    return {'x': x, 'inverse': (x-offset)/safe, 'forward': x*slope+offset,
            'b': x@bmap, 'bi': ((x-offset)/safe)@bmap, 'bf': (x*slope+offset)@bmap}


def _kernel(x, width, quadratic=0.):
    ip = x @ x.T
    return 1. + ip + 2*np.exp(-width*cdist(x,x,'sqeuclidean')) + quadratic*(ip*ip)


def _predict(K, n, target, ridge):
    dual = solve(K[:n,:n] + ridge*np.eye(n), target, assume_a='pos')
    return K[n:,:n] @ dual


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack([train['x'], study['test']['x']])
    n = len(train['x'])
    z = _coordinates(study, x)
    y = _predict(_kernel(z['bi'], .25), n, np.asarray(train['y']), .001)
    T = np.asarray(train['traj'])
    _, _, vt = np.linalg.svd(T, full_matrices=False)
    mode = vt[0]
    amp = _predict(_kernel(z['bf'], .25, .6), n, T @ mode, .001)
    residual = T - np.outer(T @ mode, mode)
    state = np.outer(amp, mode) + _predict(_kernel(z['x'], .25), n, residual, .1)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}
