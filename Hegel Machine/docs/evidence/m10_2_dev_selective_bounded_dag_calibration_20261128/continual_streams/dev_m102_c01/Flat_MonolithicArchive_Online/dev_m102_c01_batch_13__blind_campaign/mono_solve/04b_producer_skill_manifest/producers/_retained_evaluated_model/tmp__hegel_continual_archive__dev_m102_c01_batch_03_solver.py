"""Batch prediction with a calibrated spatial kernel and shared temporal modes."""
import numpy as np
from scipy.spatial.distance import cdist


def _calibration(panel):
    u, v = map(np.asarray, (panel['u'], panel['v']))
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slopes = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)
    offsets = np.median(v - u * slopes, axis=0)
    for _ in range(3):
        r = np.linalg.norm(v - u * slopes - offsets, axis=1)
        med = np.median(r)
        keep = r <= max(6 * med, 0.0005)
        if keep.sum() < 5:
            break
        for k in range(u.shape[1]):
            slopes[k], offsets[k] = np.linalg.lstsq(np.c_[u[keep, k], np.ones(keep.sum())], v[keep, k], rcond=None)[0]
    return np.where(abs(slopes) > 1e-6, slopes, 1.), offsets

from scipy.linalg import cho_factor, cho_solve

def _kernels(z):
    a=z@z.T; d=np.maximum(cdist(z,z,'sqeuclidean'),0)
    for h in (8,12,18,28):
        for w in (0,.25,1.): yield 1+w*a+np.exp(-d/h)
    for h in (6,12,22,40):
        for p in (2,3,4,5): yield (1+a/h)**p
    for h in (.2,.5,1.):
        for w in (.25,1.,3.): yield 1+a+w*a*a/4+h*a**3/16+.1*a**4/64

def _fit(k,n,y,noise):
    A=cho_solve(cho_factor(k[:n,:n]+noise*np.eye(n)),np.eye(n)); beta=A@y
    return k[n:,:n]@beta, beta/np.diag(A).reshape((-1,)+(1,)*(y.ndim-1))

def _structured_readout(z, n, y, panel):
    # Lift the inverse auxiliary coordinates using two complementary bases.
    u = np.asarray(panel['u'])
    v = np.asarray(panel['v'])
    offsets, *matrix = np.linalg.lstsq(np.c_[np.ones(len(u)), u], v, rcond=None)[0]
    w = (z - offsets) @ np.linalg.pinv(np.asarray(matrix))
    base = 1 + z @ z.T / 4 + np.exp(-cdist(z, z, 'sqeuclidean') / 4)
    f = np.arctan(1.5 * w)
    bounded = base + (1 + f @ f.T / w.shape[1])**2
    f = np.c_[np.sin(2.1 * w), np.cos(2.1 * w)]
    periodic = base + 1 + f @ f.T / w.shape[1]
    return (_fit(bounded, n, y, 1e-6)[0] + _fit(periodic, n, y, 1e-6)[0]) / 2


def solve_batch(study: dict) -> dict:
    train=study['train']; n=len(train['x']); y=np.asarray(train['y']); t=np.asarray(train['traj'])
    a,b=_calibration(study['panel_a']); z=(np.vstack((train['x'],study['test']['x']))-b)/a
    if len(z)==n: return {'y_hat':np.empty(0),'state_pred':np.empty((0,t.shape[1])),'diagnostics_vec':np.zeros(4)}
    structured = _structured_readout(z, n, y, study['panel_b'])
    z=z/max(np.sqrt(np.mean(z[:n]**2)), 1e-12); preds=[]; errs=[]
    for k in _kernels(z):
        for noise in (1e-7,1e-5,.0003,.003):
            p,r=_fit(k,n,y,noise); preds.append(p); errs.append(np.mean(abs(r)))
    errs=np.array(errs); w=np.exp(-(errs-errs.min())/.004); yh=.5*(w@preds/w.sum() + structured)
    modes=np.linalg.svd(t,full_matrices=False)[2][:2]; d=cdist(z,z,'sqeuclidean'); K=1+.25*z@z.T+np.exp(-d/18)
    th=_fit(K,n,t@modes.T,.002)[0]@modes
    return {'y_hat': yh,'state_pred': th,'diagnostics_vec':np.array([yh.mean(),yh.std(),th[:,-1].mean(),th[:,-1].std()])}
