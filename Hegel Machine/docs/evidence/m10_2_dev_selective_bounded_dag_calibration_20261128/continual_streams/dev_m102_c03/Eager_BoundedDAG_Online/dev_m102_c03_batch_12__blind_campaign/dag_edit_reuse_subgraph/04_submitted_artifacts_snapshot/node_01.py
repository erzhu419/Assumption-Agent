"""Recover calibrated panel features and fit the observable readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _coordinates(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(v, axis=1) < 5
    u, v = u[keep], v[keep]
    g = np.roll(u, 2, axis=1)

    def profile(p, coeff=False):
        f = [_bias(np.column_stack((u[:, j], np.tanh(p[0]*u[:, j]), np.sin(p[1]*g[:, j])))) for j in range(4)]
        b = np.array([_ls(a, v[:, j]) for j, a in enumerate(f)]).T
        return b if coeff else np.column_stack([a@b[:, j] for j, a in enumerate(f)]).ravel()-v.ravel()

    p = least_squares(profile, [1.8, 2.4], bounds=([.5, 1.5], [3., 3.2])).x
    b, a, c, d = profile(p, True)
    z = (x-b)/a
    ii = np.arange(4)
    for _ in range(35):
        g = np.roll(z, 2, axis=1)
        t = np.tanh(p[0]*z)
        r = b+a*z+c*t+d*np.sin(p[1]*g)-x
        if np.max(np.abs(r), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), 4, 4))
        jac[:, ii, ii] = a+c*p[0]*(1-t*t)
        jac[:, ii, (ii+2)%4] = d*p[1]*np.cos(p[1]*g)
        z -= np.linalg.solve(jac, r[..., None])[..., 0]
    return z


def _panels(panel, z):
    u, v, w = (panel[k] for k in ('u', 'v1', 'v2'))
    pair = lambda q: _bias(q*np.roll(q, 1, axis=1))
    def profile(p):
        f = _bias(np.cos(p[0]*u))
        return (f@_ls(f, w)-w).ravel()
    p = least_squares(profile, [2.4], bounds=([1.8], [3.])).x[0]
    return pair(z)@_ls(pair(u), v), _bias(np.cos(p*z))@_ls(_bias(np.cos(p*u)), w)


def _predict(study):
    tr = study['train']
    n = len(tr['x'])
    z = _coordinates(study['panel_a'], np.vstack((tr['x'], study['test']['x'])))
    a, b = _panels(study['panel_c'], z)
    f = _bias(np.column_stack((a, b, np.sum(a*b, axis=1), a[:, 0]*b[:, 1])))
    y = f[n:, :-1]@_ls(f[:n, :-1], tr['y'])
    state = f[n:]@_ls(f[:n], tr['traj'])
    return y, state


def run_node(inputs: dict) -> dict:
    y, state = _predict(inputs['__input__'])
    sparse = inputs['node_00']
    return {'panel_pred': np.column_stack((2*y-sparse['y_hat'], (state-.35*sparse['state_pred'])/.65))}
