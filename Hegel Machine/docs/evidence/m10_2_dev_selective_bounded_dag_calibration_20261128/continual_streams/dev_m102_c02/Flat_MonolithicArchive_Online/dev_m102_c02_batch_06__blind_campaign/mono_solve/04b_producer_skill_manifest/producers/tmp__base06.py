"""Robust cyclic calibration, panel-B operator, and shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    r = np.roll(u, -1, axis=1)
    d = u.shape[1]
    j = np.arange(d)
    h = np.zeros((len(u), d, 2*d+3))
    h[:, j, j] = 1
    h[:, j, d+j] = u
    h[:, :, -3:] = np.stack((r, u*r, u**3), axis=-1)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(12):
        p = _fit(h[keep].reshape(-1, h.shape[-1]), v[keep].ravel())
        error = np.linalg.norm(h @ p-v, axis=1)
        new = error <= max(5*np.median(error), 1e-8)
        if np.array_equal(keep, new) or new.sum() < 2*d+3:
            break
        keep = new
    bias, gain = p[:d], p[d:2*d]
    linear, pair, cubic = p[-3:]

    def forward(z):
        other = np.roll(z, -1)
        return bias + gain*z + linear*other + pair*z*other + cubic*z**3

    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/gain,
                                   gtol=1e-11).x for row in x])



def _warp(x, h, p):
    return np.sign(x)*np.maximum(np.abs(x)-h,0)**p

def _quad(a):
    i,j=np.triu_indices(a.shape[1]); return a[:,i]*a[:,j]

def _cross(a,b):
    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)

def _ridge(H,Y,n,reg):
    m=H[:n].mean(0); sd=np.maximum(H[:n].std(0),1e-9); G=(H-m)/sd
    ym=Y.mean(0); W=G[:n]
    return ym+G[n:]@np.linalg.solve(W.T@W+reg*np.eye(W.shape[1]),W.T@(Y-ym))

def solve_batch(study):
    tr=study['train']; n=len(tr['x']); z=_calibration(study['panel_a'],np.r_[tr['x'],study['test']['x']]); w=_warp(z,.39,1.25)
    h=np.c_[w,z,_quad(w),_quad(z),_cross(w,z)]
    y=.94*_ridge(h,tr['y'],n,.02)+.06*_ridge(np.c_[w,z,_quad(w),_cross(w,z)],tr['y'],n,.02)
    w=_warp(z,.35,1.5); T=_ridge(np.c_[z,w,_quad(z),_cross(w,z)],tr['traj'],n,.02)
    return {'y_hat':y,'state_pred':T,'diagnostics_vec':np.array([y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()])}
