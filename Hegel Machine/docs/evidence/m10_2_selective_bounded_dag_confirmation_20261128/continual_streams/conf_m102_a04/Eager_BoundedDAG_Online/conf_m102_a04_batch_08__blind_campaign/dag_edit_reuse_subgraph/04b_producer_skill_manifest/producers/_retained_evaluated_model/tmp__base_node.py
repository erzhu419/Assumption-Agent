import numpy as np
from scipy.optimize import least_squares
from itertools import combinations


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _pairs(x):
    return np.column_stack([x[:, i]*x[:, j] for i, j in combinations(range(x.shape[1]), 2)])


def _coordinates(study):
    a = study['panel_a']
    u, v = np.asarray(a['u'], float), np.asarray(a['v'], float)
    x = np.vstack((study['train']['x'], study['test']['x'])).astype(float)
    d = u.shape[1]
    coef = np.empty((5, d))
    for j in range(d):
        k = (j+1) % d
        f = np.column_stack((np.ones(len(u)), u[:, j], u[:, k], u[:, j]*u[:, k], u[:, j]**3))
        b = _fit(f, v[:, j])
        for _ in range(10):
            r = abs(f@b-v[:, j]); w = 1/np.sqrt(np.maximum(r, 2e-4))
            b = _fit(f*w[:, None], v[:, j]*w)
        r = abs(f@b-v[:, j]); keep = r <= max(8*np.median(r), .001)
        coef[:, j] = _fit(f[keep], v[keep, j]) if keep.sum() >= 5 else b
    o, a, b, c, e = coef
    z = (x-o)/a
    idx = np.arange(d)
    for _ in range(20):
        q = np.roll(z, -1, axis=1)
        r = o + a*z + b*q + c*z*q + e*z**3 - x
        if np.max(abs(r)) < 1e-10: break
        jac = np.zeros((len(z), d, d))
        jac[:, idx, idx] = a+c*q+3*e*z*z
        jac[:, idx, (idx+1)%d] = b+c*z
        z -= np.clip(np.linalg.solve(jac, r[...,None])[...,0], -.4, .4)
    p = study['panel_b']; rot = _fit(_one(p['u']), p['v'])
    return z, _one(z)@rot, (z-rot[0])@np.linalg.inv(rot[1:])


def _maps(study, xs):
    p = study['panel_c']; u = p['u']; n = u.shape[1]
    coeff = _fit(_one(_pairs(u)), p['v1'])
    def f(x, q):
        x2 = x*x
        return np.column_stack((_one(x), x2, x2/(1+q*x2)))
    v = p['v2']
    def residual(q):
        a = f(u, q)
        return (a@_fit(a,v)-v).ravel()
    q = least_squares(residual, np.full(n,.25), bounds=(.02,2), gtol=1e-12).x
    c = _fit(f(u,q),v)
    return [(_one(_pairs(w))@coeff, f(w,q)@c) for w in xs]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    n = len(study['train']['x'])
    coords = _coordinates(study)
    c, d = _maps(study, coords)[1]
    w = np.column_stack((c,d))
    return {'features': (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1), 'n_train': n}
