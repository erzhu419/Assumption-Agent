import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def aff(u): return np.c_[np.ones(len(u)),u]
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def quad(u): return np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)])

def coords(s,x):
 p=s['panel_a']; u=p['u'];v=p['v']; ii=np.linalg.norm(v-u,axis=1)<2;u=u[ii];v=v[ii]
 def ades(a,w): return aff(np.c_[a,np.sin(w*a)])
 def ar(w):
  a=ades(u,w[0]);return (a@ls(a,v)-v).ravel()
 w=least_squares(ar,[2.45],bounds=(1.5,3.3)).x[0]; c=ls(ades(u,w),v)
 z=np.linalg.solve(c[1:5].T,(x-c[0]).T).T
 for _ in range(25):
  j=c[1:5].T[None]+c[5:].T[None]*(w*np.cos(w*z))[:,None,:]
  z+=np.clip(np.linalg.solve(j,(x-ades(z,w)@c)[...,None])[...,0],-.3,.3)
 p=s['panel_b']; u=p['u'];v=p['v']
 def bd(a,p):return aff(a+p[0]*np.sin(p[1]*a))
 def br(p):
  a=bd(u,p);return (a@ls(a,v)-v).ravel()
 p=least_squares(br,[-.4,2.9],bounds=([-1,2],[0,4])).x
 return z,bd(z,p)@ls(bd(u,p),v)

def ridge(a,v,b,l):
 m=a.mean(0); sd=np.maximum(a.std(0),1e-10); a=(a-m)/sd;b=(b-m)/sd
 return (v.mean(0)+b@np.linalg.solve(a.T@a+l*np.eye(a.shape[1]), a.T@(v-v.mean(0))))

def solve_batch(study):
 t=study['train']; n=len(t['x']); z,q=coords(study,np.r_[t['x'],study['test']['x']]);
 a=np.c_[z,q,quad(z),quad(q),(z[:,:,None]*q[:,None,:]).reshape(len(z),-1)]
 yh=ridge(a[:n],t['y'],a[n:],.25); tr=ridge(a[:n],t['traj'],a[n:],.003)
 return dict(y_hat=yh,state_pred=tr,diagnostics_vec=np.array([yh.mean(),yh.std(),tr[:,-1].mean(),tr[:,-1].std()]))
