import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; T=s['train']['traj']; np.set_printoptions(precision=5,suppress=True)
Ls={'2exp':(lambda p:np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)],[2,.4]),'exp+linear':(lambda p:np.c_[-np.expm1(-p[0]*t),t],[2]),'exp+log':(lambda p:np.c_[-np.expm1(-p[0]*t),np.log1p(p[1]*t)],[2,.5]),'exp+sin':(lambda p:np.c_[-np.expm1(-p[0]*t),np.sin(p[1]*t)],[2,.5]),'exp+const':(lambda p:np.c_[-np.expm1(-p[0]*t),np.ones_like(t)],[2]),'exp+t exp':(lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)],[2,.4]),'rational2':(lambda p:np.c_[t/(1+p[0]*t),t/(1+p[1]*t)],[2,.4])}
_,_,V=np.linalg.svd(T,full_matrices=False); print('svd',V[:2]);
for k,(fun,p) in Ls.items():
 def err(p):
  b=fun(p); return (b@np.linalg.lstsq(b,T.T,rcond=None)[0]-T.T).ravel()
 opt=least_squares(err,p,bounds=(1e-5,20)); print(k,opt.x,np.sqrt(np.mean(opt.fun**2)),fun(opt.x).T)
for k,v in s['panel_c'].items():
 if k=='u':continue
 u=s['panel_c']['u']; A=np.c_[np.ones(len(u)),u]; print(k)
 for f,inv,initial in [('tanh',np.tanh,np.ones(4)*1.4),('sin',np.sin,np.ones(4)*1.5),('identity',lambda v:v,np.ones(4))]:
  for j in range(4):
   q=least_squares(lambda p:p[0]*inv(A@p[1:])-v[:,j],np.r_[initial[j],np.linalg.lstsq(A,v[:,j],rcond=None)[0]]); print(f,j,np.std(q.fun),q.x)
