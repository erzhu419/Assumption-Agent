"""Calibrated regression with separate readout and temporal response kernels."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import cho_factor, cho_solve


def fit(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-11)[0]


def affine(x):
    return np.c_[np.ones(len(x)),x]


def calibrate(panel,x):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    good=np.linalg.norm(v,axis=1)<5
    def design(u,j):
        a=u[:,j]; b=u[:,(j+1)%4]
        return np.c_[affine(u),a*b,a**3]
    co=np.column_stack([fit(design(u[good],j),v[good,j]) for j in range(4)])
    def forward(z):
        return np.column_stack([design(z,j)@co[:,j] for j in range(4)])
    g=np.linalg.solve(co[1:5].T,(x-co[0]).T).T
    return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(g,x)])



def _kernel(z,n,h,g,w,l,variant=0):
    a=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),1e-8)
    b=z-g*np.sin(h*z)
    b=(b-b[:n].mean(0))/np.maximum(b[:n].std(0),1e-8)
    G=(b@b.T+w*(a@a.T))/(4*(1+w))
    R=((b[:,None]-b[None,:])**2+w*(a[:,None]-a[None,:])**2)/(1+w)
    r=np.exp(-R/l); m=r.sum(2)
    return (1+G)**2+m/4+(m*m-(r*r).sum(2))/12

def _pred(K,Y,n,r):
    m=Y.mean(0)
    return K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+r*np.eye(n)),Y-m)+m

def solve_batch(study):
    tr=study['train']; n=len(tr['x']); z=calibrate(study['panel_a'],np.r_[tr['x'],study['test']['x']]); y=tr['y']; T=tr['traj']
    yh=np.mean([_pred(_kernel(z,n,h,g,w,l),y,n,r) for h,g,w,l,r in [(2.4,.4,.1,15,.001),(2.9,.4,.4,15,.001),(3.1,.4,.4,15,.001),(3.1,.55,1,6,.005)]],0)
    a=(z-z[:n].mean(0))/z[:n].std(0); b=z-.4*np.sin(2.9*z); b=(b-b[:n].mean(0))/b[:n].std(0)
    P=_pred((1+(b@b.T+.1*a@a.T)/4.4)**2,T,n,.001)
    return {'y_hat':yh,'state_pred':P,'diagnostics_vec':np.array([yh.mean(),yh.std(),P[:,-1].mean(),P[:,-1].std()])}
