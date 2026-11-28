"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-11)[0]


def _roll(x):
    return np.roll(x, -1, axis=1)


def _coords(study, x):
    u, v = [np.asarray(study['panel_a'][k], float) for k in ('u','v')]
    good = np.linalg.norm(v, axis=1) < 5
    u, v = u[good], v[good]
    def features(z):
        return np.stack((np.ones_like(z), z, _roll(z), z*_roll(z), z**3), axis=2)
    a = features(u)
    c = np.stack([_ls(a[:, j], v[:, j]) for j in range(4)], axis=1)
    z = np.array(x, dtype=float, copy=True)
    for _ in range(28):
        delta = np.sum(features(z)*c.T, axis=2) - x
        jac = np.eye(4)[None,:,:]*(c[1]+c[3]*_roll(z)+3*c[4]*z*z)[:,:,None]
        jac += np.roll(np.eye(4), 1, axis=1)[None,:,:]*(c[2]+c[3]*z)[:,:,None]
        z -= np.clip(np.linalg.solve(jac, delta[...,None])[...,0], -.5, .5)
    return z


def _poly(h, degree):
    return np.column_stack([np.ones(len(h))] + [np.prod(h[:, idx], axis=1) for d in range(1,degree+1) for idx in combinations_with_replacement(range(h.shape[1]),d)])


def _panel_b(study, z):
    u, v = [np.asarray(study['panel_b'][k], float) for k in ('u','v')]
    def f(x, w):
        return np.c_[np.ones(len(x)), x, np.sin(w*x)]
    w = least_squares(lambda w: (f(u,w)@_ls(f(u,w),v)-v).ravel(), [2.9], bounds=(1., 4.)).x
    return f(z,w)@_ls(f(u,w),v)


def solve_batch(study):
    tr, te = study['train'], study['test']
    n = len(tr['x'])
    z = _coords(study, np.concatenate((tr['x'], te['x']), axis=0))
    h = _panel_b(study, z)
    f = _poly(h, 2)
    pred = f[n:] @ _ls(f[:n], np.column_stack((tr['y'], tr['traj'])))
    y, t = pred[:,0], pred[:,1:]
    return {'y_hat': y, 'state_pred': t, 'diagnostics_vec': np.array([y.mean(), y.std(), t[:,-1].mean(), t[:,-1].std()])}
