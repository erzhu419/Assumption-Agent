"""Fit coordinate and response calibrations from the study panels."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _align(panel, x):
    u, v = panel['u'], panel['v']
    n, d = u.shape
    opposite = (np.arange(d) + d//2) % d
    # Screen gross row corruptions using a robust joint panel fit.
    eye = np.eye(d)[None, :, :]
    local = np.concatenate((np.broadcast_to(eye, (n, d, d)), u[:, :, None]*eye), axis=2)

    def design(q):
        return np.concatenate((local, np.sin(q[0]*u[:, opposite, None]), np.sin(q[1]*u[:, :, None])), axis=2).reshape(n*d, -1)

    q0 = np.array([2.4, np.pi])
    a = design(q0)
    initial = np.r_[q0, _lstsq(a, v.ravel())]
    robust = least_squares(lambda p: design(p[:2])@p[2:]-v.ravel(), initial,
                           loss='cauchy', f_scale=.008).x
    residual = np.linalg.norm((design(robust[:2])@robust[2:]).reshape(v.shape)-v, axis=1)
    keep = residual <= max(.012, 6*np.median(residual))
    sel = np.repeat(keep, d)
    target = v.ravel()[sel]

    def profile(q):
        f = design(q)[sel]
        return f@_lstsq(f, target)-target

    q = least_squares(profile, robust[:2], bounds=([1.5, 2.0], [3.2, 4.2]), gtol=1e-12).x
    coef = _lstsq(design(q)[sel], target)
    offset, slope = coef[:d], coef[d:2*d]
    coupling, ripple = coef[-2:]

    def forward(z):
        return offset+slope*z+coupling*np.sin(q[0]*z[..., opposite])+ripple*np.sin(q[1]*z)

    z = (x-offset)/slope
    for _ in range(28):
        err = forward(z)-x
        if np.max(np.abs(err), initial=0) < 1e-11:
            break
        j = np.arange(d)
        jac = np.zeros((len(z), d, d))
        jac[:, j, j] = slope+ripple*q[1]*np.cos(q[1]*z)
        jac[:, j, opposite] = coupling*q[0]*np.cos(q[0]*z[:, opposite])
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        z -= step/np.maximum(1., np.linalg.norm(step, axis=1, keepdims=True))
    for i in np.flatnonzero(np.max(np.abs(forward(z)-x), axis=1) > 1e-8):
        z[i] = least_squares(lambda row: forward(row)-x[i], z[i], gtol=1e-13).x
    return z


def _response(panel, z, key, extra, bounds):
    u, target = panel['u'], panel[key]

    def features(a, q):
        return np.column_stack((np.ones(len(a)), a, extra(a), np.arctan(q*a)))

    def loss(q):
        f = features(u, q)
        return np.linalg.norm(f@_lstsq(f, target)-target)

    q = minimize_scalar(loss, bounds=bounds, method='bounded', options={'xatol': 1e-10}).x
    return features(z, q)@_lstsq(features(u, q), target)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z = _align(study['panel_a'], np.vstack((study['train']['x'], study['test']['x'])))
    panel = study['panel_c']
    c = _response(panel, z, 'v1', lambda a: a**3, (.7, 2.5))
    d = _response(panel, z, 'v2', lambda a: a*np.roll(a, -1, axis=1), (.5, 2.))
    w = np.column_stack((c, d))
    return {'features': np.column_stack((z, w)), 'calibrated': w}
