import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
U,V=(S['panel_a'][k] for k in ['u','v']); I=np.max(np.abs(V),axis=1)<3; u,v=U[I],V[I]; n=len(u)
def poly(x,d=1): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def fit(z,w): return np.linalg.lstsq(z,w,rcond=None)[0]
def mae(a): return np.mean(abs(a))
def loo(z,w):
 h=np.sum(np.linalg.qr(z,mode='reduced')[0]**2,axis=1); return mae((w-z@fit(z,w))/(1-h)[:,None])
a=poly(u); rng=np.random.default_rng(100); np.set_printoptions(precision=4,suppress=True)
for d in (1,2,3): print('panel polynomial',d,mae(poly(u,d)@fit(poly(u,d),v)-v),loo(poly(u,d),v))
for x in (u,v,S['train']['x'],S['test']['x']): print('quantile',np.quantile(x,[0,.1,.5,.9,1],axis=0).round(3),'cov',np.cov(x.T).round(3))
b=fit(a,v); print('affine',b); print('affine rows',np.c_[u,v-a@b])
trials=[]
for k in range(5000):
 ix=rng.choice(n,5,replace=False); b=fit(a[ix],v[ix]); err=np.linalg.norm(v-a@b,axis=1); trials.append((np.sum(err<.035),np.median(err),b))
for j,(cnt,md,b) in enumerate(sorted(trials,key=lambda r:(-r[0],r[1]))[:2]):
 g=np.linalg.norm(v-a@b,axis=1)<.035; print('ransac',cnt,md,'params low vs high',fit(a[g],v[g]),fit(a[~g],v[~g])); print('two-affine residual',mae(v-np.where(g[:,None],a@fit(a[g],v[g]),a@fit(a[~g],v[~g])))); print('group',np.c_[u,g.astype(int)])
for typ in ('relu','tanh','logistic','hard'):
 def design(p):
  r=a@p
  if typ=='relu': return np.c_[a,np.maximum(r,0)]
  if typ=='tanh': return np.c_[a,a*np.tanh(r)[:,None]]
  if typ=='logistic': return np.c_[a,1/(1+np.exp(np.clip(-r,-60,60))) ]
  return np.c_[a,a*(r>0)[:,None]]
 def obj(p):
  z=design(p); return (z@fit(z,v)-v).ravel()
 result=[]
 for p in np.r_[np.eye(5)[1:],rng.normal(size=(45,5))]:
  zz=least_squares(obj,p,max_nfev=140) if typ!='hard' else None; p=zz.x if zz is not None else p; result.append((mae(obj(p)),p,loo(design(p),v)))
 print(typ,sorted(result,key=lambda e:e[0])[:3]); p=min(result,key=lambda e:e[0])[1]; print(fit(design(p),v))
