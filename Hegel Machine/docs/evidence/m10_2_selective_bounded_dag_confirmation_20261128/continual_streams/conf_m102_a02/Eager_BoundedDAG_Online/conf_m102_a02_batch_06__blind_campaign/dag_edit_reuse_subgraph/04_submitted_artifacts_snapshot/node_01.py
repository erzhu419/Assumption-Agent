import numpy as np
from itertools import combinations_with_replacement


def calibrate(p, x):
    u, v = p['u'], p['v']
    delta = np.max(np.abs(v - np.median(v, axis=0)), axis=1)
    good = delta < 2.8*np.median(delta)
    u, v = u[good], v[good]
    c = []
    for i in range(4):
        q, w = u[:, i], u[:, (i+1)%4]
        c.append(np.linalg.lstsq(np.c_[np.ones(len(u)), q, w*(1+q), q**3], v[:, i], rcond=None)[0])
    b, a, g, h = np.array(c).T
    z = (x-b)/a
    i = np.arange(4)
    j = (i+1)%4
    for _ in range(25):
        w = z[:, j]
        r = b+a*z+g*w*(1+z)+h*z**3-x
        jac = np.zeros((len(z), 4, 4))
        jac[:, i, i] = a+g*w+3*h*z*z
        jac[:, i, j] = g*(1+z)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step/np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return z


def poly(x, degree):
    return np.column_stack([np.prod(x[:, j], axis=1) for d in range(degree+1) for j in combinations_with_replacement(range(x.shape[1]), d)])


def run_node(inputs: dict) -> dict:
    s, raw = inputs['__input__'], inputs['node_00']
    z = calibrate(s['panel_a'], np.r_[raw['a'][:, 1:], raw['a_test'][:, 1:]])
    y = np.c_[s['train']['y'], s['train']['traj']]
    a = poly(z, 4)
    return {'test_features': a[len(y):], 'coefficients': np.linalg.lstsq(a[:len(y)], y, rcond=None)[0]}
