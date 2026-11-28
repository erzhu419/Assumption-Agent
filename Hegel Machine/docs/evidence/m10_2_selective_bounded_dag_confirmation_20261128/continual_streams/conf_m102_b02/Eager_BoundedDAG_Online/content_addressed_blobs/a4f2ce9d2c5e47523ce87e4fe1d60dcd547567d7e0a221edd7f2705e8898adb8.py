"""Calibrate observed coordinates, then form scalar and dynamical predictors."""
import numpy as np
from scipy.optimize import least_squares


def _fit(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-10)[0]


def _panel_a(study, x):
    p = study['panel_a']
    u, v = p['u'], p['v']
    freq = study['telemetry_vec'][1]

    def design(z):
        return np.c_[np.ones(len(z)), z, np.sin(freq*z), z**3]

    f = design(u)
    # Grossly corrupted measurements are far outside the calibration range.
    keep = np.max(np.abs(v - np.median(v, axis=0)), axis=1) < 3.5
    w = _fit(f[keep], v[keep])
    for _ in range(5):
        err = np.linalg.norm(f@w-v, axis=1)
        keep = err < max(.018, 4*np.median(err[keep]))
        w = _fit(f[keep], v[keep])
    a, b, c = np.split(w[1:], 3)
    z = np.linalg.solve(a.T, (x-w[0]).T).T
    for _ in range(35):
        r = design(z)@w-x
        if np.max(np.abs(r)) < 1e-10:
            break
        j = a + freq*np.cos(freq*z)[...,None]*b + 3*z[...,None]**2*c
        step = np.linalg.solve(j.transpose(0,2,1), r[...,None])[...,0]
        step /= np.maximum(1., np.max(np.abs(step),axis=1,keepdims=True)/.45)
        nxt = z-step
        norm = np.linalg.norm(r,axis=1)
        for _ in range(12):
            bad = np.linalg.norm(design(nxt)@w-x,axis=1) > norm
            if not np.any(bad):
                break
            step[bad] *= .5
            nxt[bad] = z[bad]-step[bad]
        z = nxt
    return z


def _panel_b(study, z):
    u, v = (study['panel_b'][k] for k in ('u','v'))

    def h(x,q):
        return np.c_[np.ones(len(x)), np.sign(x)*np.maximum(np.abs(x)-q[0],0)**q[1]]

    def residual(q):
        f = h(u,q)
        return (f@_fit(f,v)-v).ravel()

    q = least_squares(residual,[.35,1.5],bounds=([.15,.6],[.6,2.5])).x
    return h(z,q)@_fit(h(u,q),v)


def _head(x,n,y,degree,decay):
    dot = x@x.T/x.shape[1]
    dist = np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0)
    k = (1+dot)**degree + np.exp(-decay*dist)
    return k[n:,:n]@np.linalg.solve(k[:n,:n]+1e-6*np.eye(n),y)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']
    z = _panel_a(study,np.r_[tr['x'],study['test']['x']])
    b = _panel_b(study,z)
    sym = np.c_[.3*z+b,.3*z-b]
    n = len(tr['x'])
    y = .5*_head(np.arcsinh(sym),n,tr['y'],2,1.4)+.5*_head(sym,n,tr['y'],3,.15)
    return {'y_hat':y,'trajectory_design':np.arcsinh(np.c_[.3*z,b])}
