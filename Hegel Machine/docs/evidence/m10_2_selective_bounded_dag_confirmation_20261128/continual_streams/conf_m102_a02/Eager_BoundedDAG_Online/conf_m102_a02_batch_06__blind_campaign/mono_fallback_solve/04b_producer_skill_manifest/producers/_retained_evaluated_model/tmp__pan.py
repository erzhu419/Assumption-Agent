import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=160)
def aff(u):return np.c_[np.ones(len(u)),u]
def stats(A,v,ret=False):
 b=np.linalg.lstsq(A,v,rcond=1e-9)[0]; e=A@b-v; h=(np.linalg.svd(A,full_matrices=False)[0]**2).sum(1); return (e,b) if ret else np.mean(abs(e/np.maximum(1e-8,1-h[:,None])))
for k,n in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[k]; u=p['u']; v=p[n]; a=aff(u); rows=np.linalg.norm(stats(a,v,True)[0],axis=1)<5; u=u[rows];v=v[rows];a=aff(u)
 e,b=stats(a,v,True);print(k,n,'aff',stats(a,v),'B',b,'svB',np.linalg.svd(b[1:],compute_uv=False),'svE',np.linalg.svd(e,compute_uv=False))
 q=np.array([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]).T
 cub=np.array([u[:,i]**3 for i in range(4)]).T
 ss=[('square',u*u),('cube',cub),('cross',q),('square+cube',np.c_[u*u,cub]),('cross+cube',np.c_[q,cub]),('abs',abs(u)),('rollcross',u*np.roll(u,1,axis=1))]
 for name,F in ss:
  e,b=stats(np.c_[a,F],v,True); print(name,round(stats(np.c_[a,F],v),6),np.round(np.sqrt((e*e).mean(0)),5));
  if name in ('cube','cross','rollcross'):print(b[5:])
 for j,(name,f) in enumerate([('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('exp',np.exp),('atan',np.arctan),('rational',lambda z:z/(1+z*z)),('swish',lambda z:z/(1+np.exp(-z)))]):
  for z in (u,u@stats(a,v,True)[1][1:]):
   fun=lambda w:stats(np.c_[a,f(w*z)],v)
   grid=np.linspace(.05,5,100); vmin=min(grid,key=fun); o=minimize_scalar(fun,bounds=(max(.01,vmin-.07),vmin+.07),method='bounded'); print(name,round(o.x,4),round(o.fun,6),end='; ')
  print()
