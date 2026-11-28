import pickle, numpy as np
from scipy.optimize import minimize, minimize_scalar
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u=s['panel_b']['u'];v=s['panel_b']['v'];n=len(u)
fit=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def score(X,y=v):
 m=fit(X,y);r=y-X@m;h=np.sum(X*np.linalg.pinv(X).T,axis=1);return np.linalg.norm(r),np.linalg.norm(r/(1-h).reshape((-1,)+(1,)*(r.ndim-1))),m,r
np.set_printoptions(precision=5,suppress=True)
for ij,kl in [([0,3],[0,1]),([1,2],[2,3])]:
 x=u[:,ij];y=v[:,kl]
 for name,f in [('linear',lambda x,w:x),('sin',lambda x,w:np.sin(x*w)),('tanh',lambda x,w:np.tanh(x*w)),('rational',lambda x,w:x/(1+w*x*x)),('cube',lambda x,w:x+w*x**3)]:
  F=lambda w:np.c_[np.ones(n),f(x,w)]
  o=minimize(lambda w:score(F(w),y)[0]**2,[1.,1.],bounds=[(.03,6)]*2);e,l,m,r=score(F(o.x),y);print(ij,name,'w',o.x,'err loo',e,l,'coef',m.ravel())
D=np.c_[np.ones(n),u];pool={}
for d in (2,3):
 for ix in combinations_with_replacement(range(4),d):pool['p'+str(ix)]=np.prod(u[:,ix],axis=1)
for i,j in combinations(range(4),2):
 for k in (-1.,1.):
  z=u[:,i]+k*u[:,j]
  for w in (1.,1.5,2.,2.5,3.):pool[f'sin({w}({i}{k:+g}*{j}))']=np.sin(w*z)
for i in range(4):
 for w in (.5,1.,1.5,2.,2.5,3.,3.5,4.):
  pool[f'sin({w}x{i})']=np.sin(w*u[:,i]);pool[f'cos({w}x{i})']=np.cos(w*u[:,i]);pool[f'tanh({w}x{i})']=np.tanh(w*u[:,i]);pool[f'rat({w}x{i})']=u[:,i]/(1+w*u[:,i]**2)
for q in range(4):
 y=v[:,q];X=D.copy();avail=pool.copy();print('output',q)
 for step in range(6):
  best=min(avail,key=lambda k:score(np.c_[X,avail[k]],y)[0]);X=np.c_[X,avail.pop(best)];e,l,m,r=score(X,y);print(best,round(e,6),round(l,6),np.round(m,4))
 for inds in (range(4),[0,3] if q<2 else [1,2]):
  x=u[:,inds];F=lambda w:np.c_[np.ones(n),np.sin(x*w)];o=minimize(lambda w:score(F(w),y)[0]**2,[2.]*len(inds),bounds=[(.01,6)]*len(inds));e,l,m,r=score(F(o.x),y);print('pure sine',list(inds),o.x,e,l,m)
