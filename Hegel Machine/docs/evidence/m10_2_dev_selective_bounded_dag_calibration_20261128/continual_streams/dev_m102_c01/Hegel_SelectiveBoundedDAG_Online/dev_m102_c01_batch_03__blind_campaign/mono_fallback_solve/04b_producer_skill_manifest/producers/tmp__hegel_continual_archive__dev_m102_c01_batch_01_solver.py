"""Panel-calibrated spatial prediction with a shared trajectory response."""
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
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


def _rational_panel(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v2'))

    def features(x, width):
        xx = x*x
        return np.c_[np.ones(len(x)), x, xx/(1. + width*xx)]

    def loo_error(width):
        q, _ = np.linalg.qr(features(u, width), mode='reduced')
        residual = v - q @ (q.T @ v)
        leverage = np.sum(q*q, axis=1, keepdims=True)
        return np.mean((residual/np.maximum(1. - leverage, 1e-8))**2)

    width = minimize_scalar(loo_error, bounds=(.02, 2.), method='bounded').x
    coef = np.linalg.lstsq(features(u, width), v, rcond=None)[0]
    return lambda x: features(x, width) @ coef


def _auxiliary_kernel(op, x, n):
    z = op(x)
    z -= z[:n].mean(axis=0)
    ip = z @ z.T
    spread = 2.*np.sum(z[:n]**2)/n
    return ip + .2*ip**2 + np.exp(-.72*cdist(z,z,'sqeuclidean')/max(spread, 1e-12))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack([train['x'], study['test']['x']])
    n = len(train['x'])
    z = _coordinates(study, x)
    op = _rational_panel(study['panel_c'])
    cf = _auxiliary_kernel(op, z['bf'], n)
    ci = _auxiliary_kernel(op, z['bi'], n)
    smooth = _kernel(z['bf'], .18)
    Y = np.asarray(train['y'])
    y = .6*_predict(smooth + .5*cf, n, Y, .001) + .4*_predict(smooth + .1*cf, n, Y, .001)
    T = np.asarray(train['traj'])
    _, _, vt = np.linalg.svd(T, full_matrices=False)
    mode = vt[0]
    a = T @ mode
    poly = (z['forward'] @ z['forward'].T)**2
    k = _kernel(z['bf'], .18, .2) + .3*poly
    amp = .5*_predict(k + .5*ci, n, a, .004) + .5*_predict(k + .1*ci, n, a, .004)
    state = np.outer(amp, mode) + _predict(_kernel(z['x'], .25), n, T - np.outer(a, mode), .1)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}
