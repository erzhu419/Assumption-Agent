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

def _features(z,b,w,mode):
 G=b*np.roll(z,-1,axis=1)
 if mode==0:return np.c_[w,z,_quad(w),_quad(z),_cross(w,z)]
 if mode==1:return np.c_[w,z,_quad(w),_cross(w,z)]
 if mode==2:return np.c_[w,z,_quad(w),G]
 if mode==3:return np.c_[w,z,_quad(w),G,_quad(z)]
 if mode==4:return np.c_[w,z,_quad(w),G,G*np.roll(z,-1,axis=1)]
 if mode==5:return np.c_[w,z,_quad(w),G,_cross(w,z)]
 if mode==6:return np.c_[w,z,_quad(w),G,_quad(z),_cross(w,z)]
 if mode==7:return np.c_[w,z,_quad(w),G,b/(1+.22*np.roll(z,-1,axis=1)**2)]
 if mode==8:return np.c_[w,z,_quad(w),G,_quad(b)]
 return np.c_[w,z,G,_quad(z),_cross(w,z)]

def _operator_b(panel,z):
    u, v = panel['u'],panel['v']
    def design(p):
        return _affine(_warp(u,*p))
    def residual(p):
        a=design(p)
        return (a@_fit(a,v)-v).ravel()
    p=least_squares(residual,[.35,1.5],bounds=([0,.5],[.8,3])).x
    return _affine(_warp(z,*p))@_fit(design(p),v)


def solve_batch(study: dict) -> dict:
    tr=study['train']; n=len(tr['x'])
    z=_calibration(study['panel_a'],np.concatenate((tr['x'],study['test']['x'])))
    b=_operator_b(study['panel_b'],z)
    def predict(params,target):
        h,k,mode,reg=params
        return _ridge(_features(z,b,_warp(z,h,k),mode),target,n,reg)
    y=0.4*predict((0.345, 1.25, 7, 0.06),tr['y'])+(1-0.4)*predict((0.415, 1.2, 2, 0.02),tr['y'])
    state=predict((0.35, 1.5, 8, 0.002),tr['traj'])
    terminal=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),terminal.mean(),terminal.std()])}
