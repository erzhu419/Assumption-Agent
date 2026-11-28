"""Calibrate coordinates, apply the panel B operator, and predict both readouts."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    norms = np.linalg.norm(v, axis=1)
    keep = norms <= max(4., 3.*np.median(norms))
    u, v = u[keep], v[keep]
    other = np.roll(u, 2, axis=1)

    def warp(z, p):
        return z + p[0]*np.arctan(p[1]*z)

    def profile(p):
        w = warp(u, p)
        sin = np.sin(p[2]*other)
        w0 = w-w.mean(axis=0)
        sin0 = sin-sin.mean(axis=0)
        v0 = v-v.mean(axis=0)
        ww = (w0*w0).sum(axis=0)
        ws = (w0*sin0).sum(axis=0)/ww
        wv = (w0*v0).sum(axis=0)/ww
        sin_res = sin0-w0*ws
        cross = np.sum(sin_res*v0)/np.sum(sin_res**2)
        gain = wv-cross*ws
        bias = (v-gain*w-cross*sin).mean(axis=0)
        return bias+gain*w+cross*sin-v, (bias, gain, cross)

    p = least_squares(lambda p: profile(p)[0].ravel(), [.26,1.75,2.4],
                      bounds=([.01,.2,1.2],[2.,5.,3.6]), gtol=1e-11).x
    bias, gain, cross = profile(p)[1]
    target = (x-bias)/gain
    z = target/(1+p[0]*p[1])
    # Simultaneous Newton updates of the two independent coordinate pairs.
    for _ in range(16):
        q = np.roll(z, 2, axis=1)
        r = bias+gain*warp(z, p)+cross*np.sin(p[2]*q)-x
        a = gain*(1+p[0]*p[1]/(1+(p[1]*z)**2))
        b = cross*p[2]*np.cos(p[2]*q)
        step = (np.roll(a,2,axis=1)*r-b*np.roll(r,2,axis=1))/(a*np.roll(a,2,axis=1)-b*np.roll(b,2,axis=1))
        z -= np.clip(step, -.5, .5)
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return z


def _operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, p):
        return _aff(np.sign(x)*np.maximum(np.abs(x)-p[0], 0.)**p[1])

    def profile(p):
        h = design(u, p)
        beta = _fit(h, v)
        return (h@beta-v).ravel(), beta

    p = least_squares(lambda p: profile(p)[0], [.35,1.5],
                      bounds=([0.,.25],[.75,5.]), gtol=1e-11).x
    return design(z, p)@profile(p)[1]


def _trajectory(h, q, trajectory, times):
    beta = _fit(h, trajectory)

    def project(log_rate):
        modes = -np.expm1(-np.exp(log_rate)[:,None]*times)
        return _fit(modes.T, beta.T).T@modes

    p = least_squares(lambda p: (h@(project(p)-beta)).ravel(), np.log([.45,2.]),
                      bounds=(np.log([.02,.2]),np.log([2.5,12.]))).x
    return q@project(p)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    x = np.concatenate((train['x'],test['x']),axis=0)
    z = _calibrate(study['panel_a'],x)
    b = _operator(study['panel_b'],z)
    h = _aff(b)
    paired = np.sum(b[:,:2]*b[:,2:],axis=1)
    y_design = np.column_stack((h,paired))
    y = y_design[n:]@_fit(y_design[:n],train['y'])
    state = _trajectory(h[:n],h[n:],train['traj'],np.asarray(study['eval_times']))
    terminal = state[:,-1]
    return {'y_hat':y, 'state_pred':state,
            'diagnostics_vec':np.array([y.mean(),y.std(),terminal.mean(),terminal.std()])}
