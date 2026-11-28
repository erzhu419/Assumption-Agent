"""Calibrated spatial basis and three shared relaxation modes."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _terms(z):
    w = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)


def _decode(panel, x):
    u, v = panel['u'], panel['v']
    f = _terms(u)
    keep = np.arange(len(u))
    for _ in range(6):
        coef = np.stack([_ls(f[keep, j], v[keep, j]) for j in range(4)])
        err = np.linalg.norm(np.sum(f*coef, axis=-1)-v, axis=1)
        new = np.flatnonzero(err <= 6*np.median(err))
        if np.array_equal(new, keep):
            break
        keep = new
    return np.stack([root(lambda z: np.sum(_terms(z)*coef, axis=-1)-row,
                          row, tol=1e-10).x for row in x])


def _geometry(study, x):
    z = _decode(study['panel_a'], x)
    p = study['panel_c']
    u = p['u']
    def pairs(x):
        return x[:, [0,0,1,2]]*x[:, [1,3,2,3]]
    pa = _aff(pairs(u))
    a = _aff(pairs(z))@_ls(pa, p['v1'])
    def cosine(k, x):
        return _aff(np.cos(k*x))
    def loss(k):
        d = cosine(k, u)
        return np.sum((d@_ls(d, p['v2'])-p['v2'])**2)
    k = minimize_scalar(loss, bounds=(2,3), method='bounded').x
    b = cosine(k, z)@_ls(cosine(k, u), p['v2'])
    return z, a, b


def _design(z, a, b):
    dot = np.sum(a*b, axis=1)
    cross = a[:,0]*b[:,1]
    pairs = ((0, 1), (0, 3), (1, 2), (2, 3))
    quadratic = [z[:, i]*z[:, j] for i, j in pairs]
    cubic = [np.prod(z[:, indices], axis=1)
             for indices in combinations_with_replacement(range(4), 3)
             if any(i in indices and j in indices for i, j in pairs)]
    base = _aff(np.column_stack((b, dot, cross, z, *quadratic)))
    return base, np.column_stack(cubic)


def _time_fit(t, fitted):
    def basis(p):
        return -np.expm1(-t[:, None]*p)
    def res(p):
        b = basis(p)
        return (b@_ls(b, fitted.T)-fitted.T).ravel()
    p = least_squares(res, [.4,1.8,5.3], bounds=([.08,.8,3],[.8,3,10])).x
    return basis(p)


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    base, cubic = _design(*_geometry(study, np.vstack((train['x'], study['test']['x']))))
    design = np.column_stack((base, cubic))
    coef = _ls(design[:n], train['y'])
    y = design[n:]@coef

    # Share only the cubic contribution; each mode retains its own lower-order terms.
    spatial = np.column_stack((base, cubic@coef[base.shape[1]:]))
    weights = _ls(spatial[:n], train['traj'])
    basis = _time_fit(np.asarray(study['eval_times']), spatial[:n]@weights)
    state = spatial[n:]@_ls(basis, weights.T).T@basis.T
    end = state[:,-1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
