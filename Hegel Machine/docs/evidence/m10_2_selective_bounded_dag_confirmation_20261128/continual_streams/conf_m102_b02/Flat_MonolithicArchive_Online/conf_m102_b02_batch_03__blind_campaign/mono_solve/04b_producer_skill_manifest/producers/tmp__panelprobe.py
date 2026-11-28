import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
np.set_printoptions(precision=3,suppress=True)
for pn,keys in [('panel_a',['v']),('panel_c',['v1','v2'])]:
 p=s[pn];u=p['u'];v=np.column_stack([p[k] for k in keys]); good=np.ones(len(u),bool)
 if pn=='panel_a':good=np.linalg.norm(v,axis=1)<5
 u,v=u[good],v[good];a=np.c_[np.ones(len(u)),u]
 def check(name,d):
  c=ls(d,v);r=d@c-v;h=(d*np.linalg.pinv(d).T).sum(1);print(name,d.shape[1],np.sqrt((r*r).mean()),np.sqrt(((r/(1-h[:,None]))**2).mean()),flush=True);return c
 print(pn,len(u))
 check('affine',a)
 for k in (2,3,4,5):check('diagpoly'+str(k),np.column_stack([a]+[u**i for i in range(2,k+1)]))
 for k in (2,3):
  ij=list(combinations_with_replacement(range(4),k));d=np.c_[a,np.column_stack([np.prod(u[:,q],axis=1) for q in ij])];c=check('full'+str(k),d);print(sorted(zip(ij,np.linalg.norm(c[5:],axis=1)),key=lambda t:-t[1])[:10])
 for fn in (np.tanh,np.arcsinh,np.sin,erf,lambda z:z/(1+z*z),lambda z:np.sqrt(1+z*z),lambda z:np.log1p(z*z),lambda z:1/(1+z*z),np.cos):
  def d(q):return np.c_[a,fn(u*np.exp(q))]
  def r(q):b=d(q);return (b@ls(b,v)-v).ravel()
  f=min([least_squares(r,np.full(4,np.log(i)),bounds=(-2.5,2),max_nfev=120) for i in (.35,.8,1.4,2.5,4)],key=lambda f:f.cost)
  check(getattr(fn,'__name__','?')+str(np.exp(f.x)),d(f.x))
