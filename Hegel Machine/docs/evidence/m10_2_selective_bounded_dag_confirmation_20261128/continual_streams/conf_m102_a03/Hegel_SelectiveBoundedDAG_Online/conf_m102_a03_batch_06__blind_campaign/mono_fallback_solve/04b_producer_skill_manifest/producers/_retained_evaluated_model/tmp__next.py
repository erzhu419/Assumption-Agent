import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import *
from probe import s
import copy
x,y,t=[s['train'][k] for k in ('x','y','traj')]
z,b,n=features(s); z,b=z[:72],b[:72];
for i in [0,10,17]:
 jj=np.arange(52) if not i else np.random.default_rng(i).permutation(72)[:52]; kk=np.setdiff1d(np.arange(72),jj); idx=np.r_[jj,kk];
 for f in [0,1]:
  K=kernelparts(z[idx],b[idx],52,f)
  p=np.array([adaptive(K,y[jj],sel) for sel in ('ml','loo')]); print('adaptive',i,f,np.mean(abs(p-y[kk]),axis=1),np.mean(abs(.75*p[0]+.25*p[1]-y[kk])))
 u=copy.deepcopy(s);u['train']={k:a[jj] if k!='summary' else a for k,a in u['train'].items()};u['test']['x']=x[kk];p=solve_batch(u);print('solver',i,np.mean(abs(p['y_hat']-y[kk])),np.mean(abs(p['state_pred']-t[kk])),np.mean(abs(p['state_pred']-t[kk]),axis=0))
# Check whether B has a common scalar transfer shape in every coordinate.
u,v=[s['panel_b'][k] for k in ('u','v')]
for name,alt,h in [('sin',lambda u,h:np.sin(h[0]*u),[2.8]),('tanh',lambda u,h:np.tanh(h[0]*u),[1.4]),('atan',lambda u,h:np.arctan(h[0]*u),[2]),('rat',lambda u,h:u/(1+(h[0]*u)**2),[2]),('erf',lambda u,h:__import__('scipy').special.erf(h[0]*u),[1.4]),('sin+tan',lambda u,h:np.sin(h[0]*u)+h[2]*np.tanh(h[1]*u),[2.8,1.4,1]),('rat+sin',lambda u,h:np.sin(h[0]*u)+h[2]*u/(1+(h[1]*u)**2),[2.8,1.4,1])]:
 def aa(w): return affine(u-hh(w)*0) if False else affine(u+w[-1]*alt(u,w[:-1]));
 def er(w):
  a=aa(w); return (a@fit(a,v)-v).ravel()
 r=min((least_squares(er,h+[k],max_nfev=230) for k in [-2,-.8,-.35,.3,1]),key=lambda r:r.cost);print('branch',name,np.std(r.fun),r.x)
