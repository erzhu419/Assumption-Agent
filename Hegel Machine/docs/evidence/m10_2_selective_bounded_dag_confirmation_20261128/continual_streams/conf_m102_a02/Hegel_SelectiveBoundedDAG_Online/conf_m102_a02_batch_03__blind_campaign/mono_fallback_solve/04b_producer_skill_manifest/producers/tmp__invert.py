import numpy as np,pickle
from scipy.optimize import root
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; u,v=[s['panel_a'][k] for k in ('u','v')]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=lambda x:np.c_[np.ones(len(x)),x]
def poly(x,d):return np.c_[a(x),*[np.prod(x[:,j],axis=1) for k in range(2,d+1) for j in combinations_with_replacement(range(4),k)]]
g=np.argsort(np.linalg.norm(u-v,axis=1))[:45]; u=u[g];v=v[g]
for j in (1,2,3):
 A=poly(u,j); co=ls(A,v); print('degree',j,'A cv',np.mean(np.abs((A@co-v)/(1-np.sum(A*np.linalg.pinv(A).T,axis=1))[:,None]),0));
 for mode in ('inverse','forward','revreg'):
  if mode=='inverse':
   init=(x-co[0])@np.linalg.inv(co[1:5]); results=[root(lambda q: (poly(q[None],j)@co)[0]-xi,qi) for xi,qi in zip(x,init)]; q=np.array([r.x for r in results]); print('nfail',sum(not r.success for r in results),'range',q.min(0),q.max(0))
  elif mode=='forward':q=poly(x,j)@co
  else:q=poly(x,j)@ls(poly(v,j),u)
  for k in (1,2,3):
   F=poly(q,k); e=np.abs(F[52:]@ls(F[:52],np.c_[y,z][:52])-np.c_[y,z][52:]).mean(0); print(mode,k,e[0],e[1:].mean(),e[-1])
