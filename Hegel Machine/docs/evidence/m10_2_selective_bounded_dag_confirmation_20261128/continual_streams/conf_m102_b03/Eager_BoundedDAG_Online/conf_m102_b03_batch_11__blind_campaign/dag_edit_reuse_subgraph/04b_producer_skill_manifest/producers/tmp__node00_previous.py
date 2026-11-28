"""Fit the calibrated two-branch response and two-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _align(panel):
    u, v = panel['u'], panel['v']
    # Median of pairwise slopes, followed by inlier least squares.
    i, j = np.triu_indices(len(u), 1)
    dx, dy = u[i] - u[j], v[i] - v[j]
    result = []
    for k in range(u.shape[1]):
        valid = abs(dx[:, k]) > 1e-8
        slope = np.median(dy[valid, k] / dx[valid, k])
        r = v[:, k] - u[:, k] * slope
        r -= np.median(r)
        good = abs(r) <= max(.001, 5 * np.median(abs(r)))
        result.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    return np.asarray(result).T


def _branch(panel, x, k):
    u, v = panel['u'], panel['v'+str(k)]
    if k == 1:
        def features(z, p):
            w = p[0]*z + p[1:]
            return _ones(np.c_[z, w/(1+abs(w))])
        start = [.84, 0., 0., 0., 0.]
    else:
        def features(z, p):
            return _ones(np.c_[z, np.cos(p[0]*z+p[1:])])
        start = [1.27, 0., 0., 0., 0.]
    def residual(p):
        d = features(u, p)
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, start, max_nfev=150).x
    return features(x, p) @ _ls(features(u, p), v)




def _kernel(u):
    delta = u[:, None] - u[None, :]
    return np.exp(-np.sum(delta**2, axis=2) / 4.5)


def _fit_kernel(u, f, target, n):
    K = _kernel(u)
    L = K[:n, :n] + .003*np.eye(n)
    F = f[:n]
    v = np.linalg.solve(L, np.column_stack((target, F)))
    beta = _ls(F.T @ v[:, 1:], F.T @ v[:, 0])
    return f[n:] @ beta + K[n:, :n] @ (v[:, 0] - v[:, 1:] @ beta)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    train, test = s['train'], s['test']
    n = len(train['x'])
    x = np.vstack((train['x'], test['x']))
    a, b = _align(s['panel_a'])
    z = (x-a)/b
    c, d = (_branch(s['panel_c'], z, k) for k in (1, 2))
    yz = np.c_[z,c,d]
    yz /= np.maximum(yz[:n].std(0), 1e-8)
    f = _ones(np.c_[c,d,np.sum(c*d,axis=1)])
    y = _fit_kernel(yz, f, train['y'], n)
    c, d = (_branch(s['panel_c'], x, k) for k in (1, 2))
    cross = (d[:,:,None]*c[:,None,:]).reshape(len(x), -1)
    f = _ones(np.c_[d,c,cross])
    state = f[n:] @ _ls(f[:n],train['traj'])
    return {'y_hat': y, 'unprojected': state}
