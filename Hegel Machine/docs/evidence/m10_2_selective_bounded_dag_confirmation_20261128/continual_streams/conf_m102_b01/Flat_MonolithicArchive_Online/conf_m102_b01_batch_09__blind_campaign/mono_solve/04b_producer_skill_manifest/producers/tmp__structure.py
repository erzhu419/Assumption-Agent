from explore import s,t,x,y,P,aff,ls,poly
import numpy as np
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
a=s['panel_a']; u,v=a['u'],a['v']; rng=np.random.default_rng(11)
for j in range(4):
 i=(j+2)%4; q=u[:,j]; r=u[:,i]; print('A column',j)
 items=[('opp',np.c_[q,r,q*r,q**3,r**3,q*q,r*r,q*q*r,q*r*r]),('full2',poly(u,2)),('full2+cube',np.c_[poly(u,2),u**3]),('full2+sin',np.c_[poly(u,2),np.sin(u)]),('opptrig',np.c_[q,r,q*r,np.sin(q),np.sin(r),np.tanh(q),np.tanh(r)])]
 for name,f in items:
  d=aff(f); best=(np.inf,None)
  for _ in range(180):
   idx=rng.choice(len(u),min(len(u),d.shape[1]+2),False)
   for rep in range(4):
    w=ls(d[idx],v[idx,j]); err=abs(d@w-v[:,j]); idx=np.argsort(err)[:39]
   loss=np.mean(err[idx]**2)**.5
   if loss<best[0]:best=(loss,w)
  print(name,best)
for name,fun,g in [('exp+linear',lambda p:np.c_[1-np.exp(-p[0]*t),t],[1]),('exp+exp',lambda p:np.c_[1-np.exp(-p[0]*t),1-np.exp(-p[1]*t)],[1,.25]),('exp+rat',lambda p:np.c_[1-np.exp(-p[0]*t),t/(1+p[1]*t)],[1,.25]),('exp+sin',lambda p:np.c_[1-np.exp(-p[0]*t),np.sin(p[1]*t)],[1,.4])]:
 def res(p):
  d=fun(p);return (d@ls(d,P.T)-P.T).ravel()
 q=least_squares(res,g,bounds=(.001,5)); print('time',name,q.x,np.std(q.fun),ls(fun(q.x),P.T)[:,:10])
for key in ['v1','v2']:
 c=s['panel_c']; u,v=c['u'],c[key]; print('C',key)
 for fun in [lambda u:u,lambda u:poly(u,2),lambda u:np.c_[poly(u,2),u**3],lambda u:np.c_[u,u*u],lambda u:np.c_[u,np.sin(u),np.tanh(u)],lambda u:np.c_[poly(u,2),np.tanh(u)]]:
  d=aff(fun(u)); w=ls(d,v); h=np.sum(d*np.linalg.pinv(d).T,axis=1); print(d.shape,np.sqrt(np.mean((d@w-v)**2,axis=0)),np.mean(abs((d@w-v)/np.maximum(1-h[:,None],1e-10)),axis=0));
