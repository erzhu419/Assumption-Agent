"""Calibrated, cyclic and cosine operator readouts for the study."""
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _one(a):
    return np.column_stack((np.ones(len(a)), a))


def _coords(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(v, axis=1) < 6
    u, v = u[keep], v[keep]
    j = np.arange(4)
    nxt = (j + 1) % 4
    def terms(z):
        return .32*z[:, nxt]*(z-1) + .16*z**3
    adj = v-terms(u)
    b = np.column_stack([_ls(_one(u[:, i:i+1]), adj[:, i]) for i in j])
    z = (x-b[0])/b[1]
    for _ in range(30):
        h = terms(z)+b[0]+b[1]*z-x
        A = np.zeros((len(z), 4, 4))
        A[:, j, j] = b[1]+.32*z[:, nxt]+.48*z*z
        A[:, j, nxt] = .32*(z-1)
        step = np.linalg.solve(A, h[..., None])[..., 0]
        z -= step / np.maximum(1, np.linalg.norm(step, axis=1, keepdims=True))
        if np.max(np.abs(h), initial=0) < 1e-10:
            break
    for i in np.flatnonzero(np.max(np.abs(terms(z)+b[0]+b[1]*z-x), axis=1)>1e-7):
        z[i] = least_squares(lambda h: (terms(h[None])+b[0]+b[1]*h-x[i]).ravel(), (x[i]-b[0])/b[1]).x
    return z


def _panels(panel, z):
    u = panel['u']
    def cyclic(q):
        return _one(q*np.roll(q, -1, axis=1))
    def cosine(q):
        return _one(np.cos(2.4*q))
    return cyclic(z)@_ls(cyclic(u), panel['v1']), cosine(z)@_ls(cosine(u), panel['v2'])


def _readout(z, c, d):
    return _one(np.column_stack((z*np.roll(c, 1, axis=1), c*d, c, d)))


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['y'])
    z = _coords(study['panel_a'], np.vstack((tr['x'], study['test']['x'])))
    c, d = _panels(study['panel_c'], z)
    base = _readout(z, c, d)
    # The scalar readout and state share the core operators, but the state
    # has an additional off-diagonal interaction between the two panels.
    y = base[n:] @ _ls(base[:n], tr['y'])
    state_design = np.column_stack((base, c[:, 0]*d[:, 1]))
    state = state_design[n:] @ _ls(state_design[:n], tr['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
