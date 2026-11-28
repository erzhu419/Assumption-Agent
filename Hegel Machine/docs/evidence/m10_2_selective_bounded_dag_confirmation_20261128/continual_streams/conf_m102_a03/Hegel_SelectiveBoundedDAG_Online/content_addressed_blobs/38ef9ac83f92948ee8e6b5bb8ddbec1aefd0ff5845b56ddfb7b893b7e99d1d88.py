"""Calibrated bilinear readout and a two-rate exponential state model."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    nxt = np.roll(u, -1, axis=1)
    terms = [np.column_stack((np.ones(len(u)), u[:, j], nxt[:, j],
                              u[:, j] * nxt[:, j], u[:, j] ** 3))
             for j in range(u.shape[1])]
    w = np.ones(len(u))
    for _ in range(30):
        c = np.column_stack([_fit(d * w[:, None], v[:, j] * w)
                             for j, d in enumerate(terms)])
        r = np.linalg.norm(np.column_stack([d @ c[:, j] for j, d in enumerate(terms)]) - v, axis=1)
        w = np.sqrt(np.minimum(1., 1.5 * max(np.median(r), 1e-12) / np.maximum(r, 1e-12)))
    good = r <= 6 * max(np.median(r), 1e-12)
    return np.column_stack([_fit(d[good], v[good, j]) for j, d in enumerate(terms)])


def _inverse(x, c):
    idx = np.arange(c.shape[1])
    following = (idx + 1) % len(idx)
    linear = np.diag(c[1])
    linear[idx, following] = c[2]

    def model(q):
        return c[0] + c[1] * q + c[2] * q[following] + c[3] * q * q[following] + c[4] * q**3

    def jac(q):
        j = np.diag(c[1] + c[3] * q[following] + 3 * c[4] * q**2)
        j[idx, following] = c[2] + c[3] * q
        return j

    starts = np.linalg.solve(linear, (x - c[0]).T).T
    out = np.empty_like(starts)
    for i, (row, start) in enumerate(zip(x, starts)):
        result = least_squares(lambda q: model(q) - row, start, jac=jac, gtol=1e-12)
        out[i] = result.x
    return out



def _poly(x,d):
    from itertools import combinations_with_replacement
    return np.column_stack([np.prod(x[:,p],axis=1) for k in range(1,d+1) for p in combinations_with_replacement(range(x.shape[1]),k)])


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; n=len(s['train']['x'])
    z=_inverse(np.vstack([s['train']['x'],s['test']['x']]),_calibration(s['panel_a']))
    u,v=(s['panel_b'][k] for k in ('u','v'))
    # Profile the shared saturation scale on the reference panel only.
    def design(a, log_scale):
        return np.c_[np.ones(len(a)), a, a**3/(1. + np.exp(log_scale)*a*a)]

    def residual(h):
        X = design(u, h[0])
        return (X @ _fit(X, v) - v).ravel()

    h = least_squares(residual, [-2.], bounds=(-5., 5.), max_nfev=90).x[0]
    X = design(u, h)
    g = design(z, h) @ _fit(X, v)
    i,j=np.triu_indices(g.shape[1]); p=_poly(z,3)
    c=s['panel_c']; cu=c['u']; C=_fit(np.c_[np.ones(len(cu)),cu,cu**2],c['v2']); h=np.c_[np.ones(len(z)),z,z**2]@C
    a=np.c_[g,g[:,i]*g[:,j],p,h]; b=np.c_[g,p,h]
    return {'readout_train':a[:n], 'readout_test':a[n:], 'z':b[:n], 'zt':b[n:]}
