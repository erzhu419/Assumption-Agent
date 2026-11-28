"""Calibrated operator features for the scalar and temporal readouts."""
import numpy as np
from scipy.optimize import root, minimize_scalar
from scipy.linalg import solve


def _lin(z):
    return np.column_stack((np.ones(len(z)), z))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _cal(p, points):
    u, v = (np.asarray(p[k], float) for k in ('u', 'v'))
    w = np.roll(u, -1, axis=1)
    d = np.c_[_lin(u), u*w, u**3]
    error = np.linalg.norm(v-d@_ls(d,v),axis=1)
    keep = np.argsort(error)[:-3]
    coeff = np.array([_ls(np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3][keep],v[keep,j]) for j in range(4)])
    def forward(z):
        w = np.roll(z,-1)
        return coeff[:,0]+coeff[:,1]*z+coeff[:,2]*w+coeff[:,3]*z*w+coeff[:,4]*z**3
    return np.array([root(lambda z:forward(z)-x,x).x for x in points])


def _op(p,k,z,mode):
    u,v = np.asarray(p['u'],float),np.asarray(p[k],float)
    def design(x,a):
        if mode=='b':
            return np.c_[_lin(x/(1+a*x*x)),x]
        return np.c_[_lin(np.arctan(a*x)),x,x**3]
    def error(a):
        d=design(u,a)
        h=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1)
        return np.mean(np.abs((v-d@_ls(d,v))/(1-h[:,None])))
    a=minimize_scalar(error,bounds=(.15,4),method='bounded').x
    return design(z,a)@_ls(design(u,a),v)


def _cross(a,b):
    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)


def _ridge(a,y,q,r):
    scale=np.maximum(a.std(0),1e-4)
    scale[0]=1
    p=np.eye(a.shape[1]);p[0,0]=0
    return (q/scale)@_ls(np.r_[a/scale,r*p],np.r_[y,np.zeros((a.shape[1],)+y.shape[1:])])


def _kernel(f, b, y, n, ell, reg):
    b=b/np.maximum(b[:n].std(0),1e-8)
    k=np.exp(-np.mean((b[:,None]-b[None,:n])**2,axis=2)/(2*ell**2))
    a=f[:n]; m=a.shape[1]
    g=solve(k[:n]+reg*np.eye(n),np.c_[a,y],assume_a='pos')
    beta=_ls(a.T@g[:,:m],a.T@g[:,m:])
    return f[n:]@beta+k[n:]@(g[:,m:]-g[:,:m]@beta)


def solve_batch(study: dict) -> dict:
    tr=study['train']; x=np.asarray(tr['x'],float); q=np.asarray(study['test']['x'],float); n=len(x)
    z=_cal(study['panel_a'],np.r_[x,q]); b=_op(study['panel_b'],'v',z,'b'); c=_op(study['panel_c'],'v1',z,'c')
    f=_lin(np.c_[b,c,_cross(b,c)])
    y=np.asarray(tr['y'],float)[:,None]
    y_hat=(.75*_kernel(f,b,y,n,2.,.001)+.25*_kernel(f,b,y,n,1.,.03)).ravel()
    sf=_lin(np.c_[z,b,_cross(z,b)])
    state=_ridge(sf[:n],np.asarray(tr['traj'],float),sf[n:],.08)
    return {'y_hat':y_hat,'state_pred':state,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),state[:,-1].mean(),state[:,-1].std()])}
