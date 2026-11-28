"""Calibrate the shared coordinates, then fit separate scalar and state readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, upstream, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    nxt = np.roll(u, -1, axis=1)
    designs = [np.c_[np.ones(len(u)), u[:, j], nxt[:, j], u[:, j]*nxt[:, j], u[:, j]**3] for j in range(u.shape[1])]
    gain = np.asarray(upstream['gain'])
    offset = np.asarray(upstream['offset'])
    safe_gain = np.where(np.abs(gain) > .1, gain, 1.)
    dist = np.linalg.norm(v - u, axis=1)
    m = np.median(dist)
    keep = dist <= m + max(1., 8*np.median(abs(dist-m)))
    if keep.sum() < 8:
        keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        coef = np.array([_ls(f[keep], v[keep,j]) for j,f in enumerate(designs)]).T
        err = np.linalg.norm(np.column_stack([f@coef[:,j] for j,f in enumerate(designs)]) - v, axis=1)
        med = np.median(err[keep])
        new = err <= med + max(.001, 8*np.median(abs(err[keep]-med)))
        if new.sum() < 8 or np.array_equal(new, keep):
            break
        keep = new
    bias, linear, cross, product, cubic = coef
    def forward(q):
        return bias + linear*q + (cross+product*q)*np.roll(q,-1) + cubic*q**3
    result = []
    matrix = np.diag(linear) + np.roll(np.diag(cross), 1, axis=1)
    for row in x:
        def residual(q):
            return forward(q)-row
        start = (row-offset)/safe_gain
        fit = root(residual, start, tol=1e-10)
        if np.linalg.norm(residual(fit.x)) > 1e-7:
            other = root(residual, _ls(matrix,row-bias), tol=1e-10)
            if np.linalg.norm(residual(other.x)) < np.linalg.norm(residual(fit.x)):
                fit = other
        result.append(fit.x)
    return np.asarray(result)


def _panel_b(panel, q):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    def features(x, freq):
        return np.c_[np.ones(len(x)),np.cos(freq*x)]
    def error(freq):
        a = features(u,freq)
        return np.sum((a@_ls(a,v)-v)**2)
    grid = np.linspace(.2, 9., 45)
    j = np.argmin([error(freq) for freq in grid])
    freq = minimize_scalar(error, bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]), method='bounded', options={'xatol':1e-9}).x
    a = features(u,freq)
    return features(q,freq)@_ls(a,v)


def _scalar(q, b, y):
    n = len(y)
    h = np.c_[np.ones(len(q)), b]
    quad = np.c_[h, *[b[:,i]*b[:,j] for i,j in combinations_with_replacement(range(b.shape[1]),2)]]
    z = q[:,0]**2-q[:,2]**2
    def design(k):
        return np.c_[quad, h/(1.+k*z[:,None]**2)]
    def error(k):
        a = design(k)[:n]
        return np.sum((a@_ls(a,y)-y)**2)
    k = minimize_scalar(error, bounds=(.05,10.), method='bounded').x
    a = design(k)
    return a[n:]@_ls(a[:n],y)


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    q = _calibrate(study['panel_a'], upstream, np.vstack([train['x'],study['test']['x']]))
    b = _panel_b(study['panel_b'],q)
    y = _scalar(q,b,train['y'])
    f = np.c_[np.ones(len(q)),b]
    state = f[n:]@_ls(f[:n],train['traj'])
    end = state[:,-1]
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(),y.std(),end.mean(),end.std()])}
