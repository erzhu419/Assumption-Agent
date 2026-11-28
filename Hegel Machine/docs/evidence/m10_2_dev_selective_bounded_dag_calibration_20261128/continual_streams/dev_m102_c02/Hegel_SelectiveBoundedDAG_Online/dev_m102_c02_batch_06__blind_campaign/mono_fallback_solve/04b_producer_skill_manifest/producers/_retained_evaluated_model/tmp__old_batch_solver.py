"""Calibrated nonlinear predictor for the batch readout and state trajectory."""
import numpy as np
from scipy.linalg import solve

def _calibrate(panel, x):
    u,v=(np.asarray(panel[k],float) for k in ('u','v'))
    def features(a,b): return np.column_stack((np.ones(len(a)),a,b,a*b,a**3))
    g=np.ones(len(u),bool)
    for _ in range(4):
        m=np.array([np.linalg.lstsq(features(u[g,j],u[g,(j+1)%4]),v[g,j],rcond=None)[0] for j in range(4)])
        err=np.linalg.norm(np.column_stack([features(u[:,j],u[:,(j+1)%4])@m[j] for j in range(4)])-v,axis=1)
        g=err<=np.partition(err,44)[44]
    z=np.array(x,float,copy=True); j=np.arange(4)
    for _ in range(14):
        w=np.roll(z,-1,axis=1); J=np.zeros((len(z),4,4)); J[:,j,j]=m[:,1]+m[:,3]*w+3*m[:,4]*z*z; J[:,j,(j+1)%4]=m[:,2]+m[:,3]*z
        f=m[:,0]+m[:,1]*z+m[:,2]*w+m[:,3]*z*w+m[:,4]*z**3
        z-=np.linalg.solve(J,(f-x)[...,None])[...,0]
    return z

def _predict(study,x,n):
    train=study['train']; y=np.c_[train['y'],train['traj']]; i,j=np.triu_indices(4); results=[]
    for k in (1.45,1.65,1.85):
        z=np.tanh(k*x); p=np.column_stack((np.ones(len(x)),x,z)); q=np.column_stack((z[:,i]*z[:,j],(z[:,:,None]*x[:,None,:]).reshape(-1,16)))
        p=np.c_[p,q]; w=p[:n]; results.append(p[n:]@solve(w.T@w+.001*np.eye(w.shape[1]),w.T@y,assume_a='pos'))
    return np.mean(results,axis=0)

def solve_batch(study: dict) -> dict:
    train=study['train']; n=len(train['x']); x=_calibrate(study['panel_a'],np.r_[train['x'],study['test']['x']]); out=_predict(study,x,n); y,p=out[:,0].copy(),out[:,1:].copy()
    return {'y_hat':y,'state_pred':p,'diagnostics_vec':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}
