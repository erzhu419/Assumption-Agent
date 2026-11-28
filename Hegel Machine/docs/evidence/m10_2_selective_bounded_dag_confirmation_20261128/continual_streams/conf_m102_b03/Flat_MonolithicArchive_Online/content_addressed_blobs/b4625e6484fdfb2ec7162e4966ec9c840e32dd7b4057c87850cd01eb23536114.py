import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement

def _ls(x,y): return np.linalg.lstsq(x,y,rcond=1e-9)[0]
def _one(x): return np.column_stack((np.ones(len(x)),x))
def _align(panel):
    pars=[]
    for u,v in zip(panel['u'].T,panel['v'].T):
        i,j=np.triu_indices(len(u),1); d=u[i]-u[j]; good=abs(d)>1e-10
        m=np.median((v[i]-v[j])[good]/d[good]); r=v-m*u
        good=abs(r-np.median(r))<max(.001,6*np.median(abs(r-np.median(r))))
        pars.append(_ls(_one(u[good]),v[good]))
    return np.asarray(pars).T

def _poly(x,k): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ij],axis=1) for j in range(1,k+1) for ij in combinations_with_replacement(range(x.shape[1]),j)])

def _ridge(d,y,lam):
    n=len(y); scale=np.maximum(d[:n].std(axis=0),.05); scale[0]=1
    p=d/scale; a=p[:n]; pen=np.eye(a.shape[1])*lam; pen[0,0]=0
    return p@np.linalg.solve(a.T@a+pen,a.T@y)

def _predict(study,rows,xnew):
    tr=study['train']; n=len(tr['x'][rows]); off,sc=_align(study['panel_a']); z=(np.vstack((tr['x'][rows],xnew))-off)/sc
    y=tr['y'][rows]; traj=tr['traj'][rows]; t=study['eval_times']
    h=-np.expm1(-t[:,None]*np.array([.38,1.8,5.4])); amp=_ls(h,traj.T).T
    p3=_poly(z,3); p2=p3[:,:15]; c=study['panel_c']; u=c['u']; pairs=[(0,1),(0,3),(1,2),(2,3)]
    def cross(w): return _one(np.column_stack([w[:,i]*w[:,j] for i,j in pairs]))
    ca=cross(z)@_ls(cross(u),c['v1']); co=np.cos(2.4*z); w=z/(1+1.8*abs(z)); prod=(ca[:,:,None]*w[:,None,:]).reshape(len(z),-1)
    py=_ridge(np.c_[p2,w,prod],y,.35)
    pa=_ridge(np.c_[p3,co],amp[:,0],1.)
    pb=_ridge(np.c_[p2,co,(z[:,:,None]*co[:,None,:]).reshape(len(z),-1)],amp[:,1],.4)
    pc=_ridge(p3,amp[:,2],1.0)
    return py[n:],np.column_stack((pa,pb,pc))[n:]@h.T

def solve_batch(study: dict) -> dict:
    y,state=_predict(study,slice(None),study['test']['x']); end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
