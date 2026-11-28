import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(321)
def mono(x,d):return np.column_stack([np.prod(x[:,c],axis=1) for c in itertools.combinations_with_replacement(range(x.shape[1]),d)])
def basis(x,k):
 a=np.c_[np.ones(len(x)),x]; q=mono(x,2); c=mono(x,3); cyc=np.roll(x,-1,axis=1)
 return dict(aff=a,cross=np.c_[a,x*cyc],quadratic=np.c_[a,q],cubicdiag=np.c_[a,x**3],quad_cubicdiag=np.c_[a,q,x**3],cubic=np.c_[a,q,c],sine=np.c_[a,np.sin(2*x)],cosine=np.c_[a,np.cos(2*x)],tan=np.c_[a,np.tanh(2*x)],pqsin=np.c_[a,q,np.sin(2*x)],pqcos=np.c_[a,q,np.cos(2*x)])[k]
keys='aff cross quadratic cubicdiag quad_cubicdiag cubic sine cosine tan pqsin pqcos'.split()
def fit_rob(a,v):
 n,m=a.shape; best=(np.inf,None)
 if m+5>=n: return np.linalg.lstsq(a,v,rcond=None)[0]
 for it in range(140):
  ix=rng.choice(n,m+1,replace=False); c=np.linalg.lstsq(a[ix],v[ix],rcond=None)[0]
  for _ in range(3):
   e=np.linalg.norm(a@c-v,axis=1); keep=e.argsort()[:int(.83*n)];c=np.linalg.lstsq(a[keep],v[keep],rcond=None)[0]
  score=np.median(np.linalg.norm(a@c-v,axis=1));
  if score<best[0]:best=(score,c)
 return best[1]
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; print(p,'range',np.round([u.min(),u.max()],3))
 for j,v in s[p].items():
  if j=='u':continue
  for k in keys:
   a=basis(u,k)
   if a.shape[1]>len(u)-4:continue
   if p=='panel_a':
    w=fit_rob(a,v); e=np.linalg.norm(a@w-v,axis=1); keep=e.argsort()[:40]
   else:keep=np.arange(len(u))
   a=a[keep]; y=v[keep]; w=np.linalg.lstsq(a,y,rcond=None)[0]; e=a@w-y; h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,1)
   print(j,k,'rmse/loo',round(np.mean(e**2)**.5,6),round(np.mean((e/(1-h[:,None]))**2)**.5,6))
   if k=='aff': print(np.round(w,5))
T=s['train']['traj'].T;t=s['eval_times']
def b(r):return -np.expm1(-t[:,None]*r)
def err(r):
 A=b(r);return (A@np.linalg.lstsq(A,T,rcond=None)[0]-T).ravel()
r=least_squares(err,[1.7,.25],bounds=([.6,.02],[6,.65])).x; print('rates',r,'rmse',np.mean(err(r)**2)**.5)
