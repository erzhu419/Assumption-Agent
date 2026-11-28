import numpy as np,pickle
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(2026)
def one(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def inter(x):return np.stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)],1)
a=s['panel_a']; u,v=a['u'],a['v']; F=one(u); best=None
for ix in rng.integers(0,len(u),size=(400,5)):
 if len(set(ix))<5:continue
 b=fit(F[ix],v[ix]); r=np.linalg.norm(F@b-v,axis=1); key=np.median(r)
 if best is None or key<best[0]:best=(key,b)
b=best[1]
for _ in range(5):
 r=np.linalg.norm(F@b-v,axis=1); ok=r<max(6*np.median(r),.0006); b=fit(F[ok],v[ok])
print('alignment',b,'kept',sum(ok),'residual quantiles',np.quantile(r,[0,.5,.8,.9,1]),flush=True)
B=fit(one(s['panel_b']['u']),s['panel_b']['v']); print('rotation',B,'singular values',np.linalg.svd(B[1:],compute_uv=False)); z=(s['train']['x']-b[0])@np.linalg.inv(b[1:]); w=one(z)@B
p=s['panel_c']; u=p['u']; print('panel C u range',u.min(0),u.max(0)); print('raw canon rotated extents',[(x.min(0),x.max(0)) for x in [z,w]])
from scipy.optimize import least_squares
for label,T in [('raw',lambda x:x),('rotated',lambda x:one(x)@B),('inverse',lambda x:(x-B[0])@np.linalg.inv(B[1:]))]:
 q=T(u); X=T(z)
 for name,fn in [('linear',one),('pairs',lambda x:one(inter(x))),('full2',lambda x:np.c_[one(x),x*x,inter(x)])]:
  f=fn(q); print(label,name,np.sqrt(np.mean((f@fit(f,p['v1'])-p['v1'])**2)))
 def d(x,r):return np.c_[one(x),x*x,x*x/(1+r*x*x)]
 def res(r):
  f=d(q,r);return (f@fit(f,p['v2'])-p['v2']).ravel()
 opt=least_squares(res,[.25],bounds=(.02,2),gtol=1e-13); print(label,'curvature',opt.x,np.sqrt(np.mean(opt.fun**2)))
