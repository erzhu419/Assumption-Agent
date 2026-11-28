"""Calibrated panel-B prediction of scalar and trajectory readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar


def _least(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibration_design(u, j):
    k = (j + 1) % u.shape[1]
    a, b = u[:, j], u[:, k]
    return np.column_stack((np.ones(len(u)), a, b, a*b, a**3))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    designs = [_calibration_design(u, j) for j in range(u.shape[1])]
    distance = np.linalg.norm(v-u, axis=1)
    keep = distance < max(5*np.median(distance), 1e-8)
    for _ in range(4):
        w = np.column_stack([_least(g[keep], v[keep, j]) for j, g in enumerate(designs)])
        err = np.linalg.norm(np.column_stack([g @ w[:, j] for j, g in enumerate(designs)])-v, axis=1)
        keep = err < max(6*np.median(err), 1e-8)
    return w


def _inverse(x, w):
    j = np.arange(x.shape[1]); k = np.roll(j, -1)
    linear = np.diag(w[1]); linear[j, k] = w[2]
    u = np.linalg.solve(linear, (x-w[0]).T).T
    jac = np.zeros((len(u), len(j), len(j)))
    for _ in range(40):
        nxt = u[:, k]
        r = w[0] + w[1]*u + w[2]*nxt + w[3]*u*nxt + w[4]*u**3 - x
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac[:, j, j] = w[1] + w[3]*nxt + 3*w[4]*u**2
        jac[:, j, k] = w[2] + w[3]*u
        du = np.linalg.solve(jac, r[..., None])[..., 0]
        u -= du/np.maximum(1., np.max(np.abs(du), axis=1, keepdims=True))
    return u


def _operator(panel, u):
    def features(x, k):
        return np.column_stack((_affine(x), x/(1+k*x*x)))
    x, y = panel['u'], panel['v']
    def score(k):
        a=features(x,k)
        h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1)
        return np.mean(np.abs((a@_least(a,y)-y)/(1-h[:,None])))
    k=minimize_scalar(score,bounds=(.25,1.6),method='bounded').x
    return features(u,k)@_least(features(x,k),y)


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2))))


def solve_batch(study: dict) -> dict:
    tr=study['train']; n=len(tr['x'])
    x=np.concatenate((tr['x'],study['test']['x']))
    u=_inverse(x,_calibrate(study['panel_a']))
    g=_operator(study['panel_b'],u)
    a,b=np.split(_quadratic(g),[n]); scale=np.maximum(.01,np.sqrt(np.mean(a*a,axis=0)))
    a,b=a/scale,b/scale
    y=b@np.linalg.solve(a.T@a+.03*np.eye(a.shape[1]),a.T@tr['y'])
    t=tr['traj']; basis=np.linalg.svd(t,full_matrices=False)[2][:2]; t=t@basis.T
    z=(b@np.linalg.solve(a.T@a+.03*np.eye(a.shape[1]),a.T@t))@basis
    return {'y_hat': y,'state_pred':z,'diagnostics_vec':np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}
