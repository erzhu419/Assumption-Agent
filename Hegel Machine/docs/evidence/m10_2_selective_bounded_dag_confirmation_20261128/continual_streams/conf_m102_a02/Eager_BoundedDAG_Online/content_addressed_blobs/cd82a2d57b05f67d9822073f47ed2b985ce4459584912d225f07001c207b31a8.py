"""Fit paired readouts in robustly calibrated panel coordinates."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    scale = np.empty(u.shape[1])
    offset = np.empty_like(scale)
    for k, (a, b) in enumerate(zip(u.T, v.T)):
        du = a[j] - a[i]
        ok = np.abs(du) > 1e-8
        m = np.median((b[j[ok]] - b[i[ok]]) / du[ok])
        r = b - m*a
        r -= np.median(r)
        keep = np.abs(r) <= max(9*np.median(np.abs(r)), 1e-6)
        offset[k], scale[k] = _fit(np.c_[np.ones(keep.sum()), a[keep]], b[keep])
    return (x-offset)/scale


def _panel_map(panel, key, f, bounds, linear):
    u, v = (np.asarray(panel[k], float) for k in ('u', key))
    def design(x, w):
        if linear:
            return np.c_[np.ones(len(x)), x, f(w*x)]
        return np.c_[np.ones(len(x)), f(w*x)]
    def loss(w):
        a = design(u, w)
        return np.mean((a @ _fit(a,v) - v)**2)
    grid = np.linspace(*bounds, 81)
    k = np.argmin([loss(w) for w in grid])
    w = minimize_scalar(loss, bounds=(grid[max(k-1,0)],grid[min(k+1,len(grid)-1)]), method='bounded', options={'xatol':1e-10}).x
    c = _fit(design(u,w),v)
    return lambda x: design(x,w) @ c


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    g = _panel_map(study['panel_b'], 'v', np.cos, (0.4,5.0), False)(z)
    p = _panel_map(study['panel_c'], 'v1', np.tanh, (0.4,4.0), True)(g)
    q = _panel_map(study['panel_c'], 'v2', lambda a: a*a/(1+a*a), (0.15,1.5), True)(g)
    quadratic = np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(g.shape[1]),2)])
    return np.c_[np.ones(len(g)), g, quadratic, p, q]


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    train = study['train']
    a, b = raw['a'], raw['a_test']
    m = _features(study, np.concatenate((a[:,1:], b[:,1:]),axis=0))
    y, t = np.asarray(train['y']), np.asarray(train['traj'])
    modes = np.linalg.svd(t, full_matrices=False)[2][:2]
    targets = np.c_[y, t @ modes.T]
    c = _fit(m[:len(a)], targets)
    return {'test_features': m[len(a):], 'coefficients': np.c_[c[:,0],c[:,1:] @ modes]}
