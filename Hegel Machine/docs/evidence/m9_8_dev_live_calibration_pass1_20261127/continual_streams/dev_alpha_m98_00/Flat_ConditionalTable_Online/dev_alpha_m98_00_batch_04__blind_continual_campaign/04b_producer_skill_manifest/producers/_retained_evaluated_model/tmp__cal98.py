import numpy as np
from itertools import product


def fit_bridge(data):
    u, raw = data['true_channel_loads'], data['bridge_raw']
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    def basis(v):
        return np.concatenate((np.ones_like(v[..., :1]), v, v**3,
                               np.stack([v[..., i]*v[..., j] for i,j in pairs], axis=-1)), axis=-1)
    a = basis(u)
    w = np.ones(len(u))
    for _ in range(14):
        c = np.linalg.lstsq(a*w[:,None], raw*w[:,None], rcond=None)[0]
        res = np.linalg.norm(a@c-raw, axis=1)
        w = np.sqrt(np.minimum(1., max(.001, 1.5*np.median(res))/np.maximum(res,1e-12)))
    keep = res < max(.005, 6*np.median(res))
    c = np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]
    return c, pairs, basis, keep


def invert_bridge(raw, cal):
    c, pairs, basis, _ = cal
    lin_inv = np.linalg.inv(c[1:5])
    u = (raw-c[0])@lin_inv
    for _ in range(15):
        err = basis(u)@c-raw
        if np.max(np.abs(err)) < 1e-10:
            break
        jac = c[1:5] + 3*u[..., :,None]**2*c[5:9]
        for j,(p,q) in enumerate(pairs):
            jac[...,p,:] += u[...,q,None]*c[9+j]
            jac[...,q,:] += u[...,p,None]*c[9+j]
        du = np.linalg.solve(jac.swapaxes(-2,-1),err[...,None])[...,0]
        u -= du / np.maximum(1.,np.max(np.abs(du),axis=-1,keepdims=True))
    return u
