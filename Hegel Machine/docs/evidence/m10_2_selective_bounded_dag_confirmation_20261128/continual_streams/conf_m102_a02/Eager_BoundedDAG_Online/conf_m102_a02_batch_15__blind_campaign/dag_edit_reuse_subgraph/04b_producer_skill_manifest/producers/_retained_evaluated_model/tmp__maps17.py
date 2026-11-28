import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=130)
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def a_design(u,shift,kind):
 w=np.roll(u,-shift,axis=1); cols=[np.ones_like(u),u,w,u*w,u**3];
 if kind>=1: cols += [u*u,w*w,w**3];
 if kind>=2: cols += [u*u*w,u*w*w];
 return np.stack(cols,axis=2)
u,v=map(s['panel_a'].get,('u','v'))
for shift in (1,2,3):
 for kind in range(3):
  a=a_design(u,shift,kind); rng=np.random.default_rng(17); best=None
  for j in range(200):
   idx=rng.choice(len(u),max(13,a.shape[-1]+2),replace=False); c=np.array([fit(a[idx,i],v[idx,i]) for i in range(4)]); e=np.linalg.norm(np.einsum('nip,ip->ni',a,c)-v,axis=1); score=np.sort(e)[:36].mean();
   if best is None or score<best[0]:best=(score,e)
  ids=np.argsort(best[1])[:40]; c=np.array([fit(a[ids,i],v[ids,i]) for i in range(4)]); e=np.linalg.norm(np.einsum('nip,ip->ni',a,c)-v,axis=1); print('A',shift,kind,np.sort(e)[[19,35,39,40,41,42,43,44,45,46,47]]); print(c if shift==2 else '')
for p in ('panel_b','panel_c'):
 u=s[p]['u']; a=np.column_stack([np.ones(len(u)),u]);
 for k,v in s[p].items():
  if k=='u':continue
  print(p,k,'linear rms',np.mean((v-a@fit(a,v))**2)**.5)
  for f in (np.cos,np.sin,erf,np.tanh):
   for transform in (lambda z:z,lambda z:z-np.roll(z,2,axis=1),lambda z:z+np.roll(z,2,axis=1),lambda z:z-np.roll(z,-1,axis=1),lambda z:z+np.roll(z,-1,axis=1)):
    b=transform(u)
    def err(w):
     d=np.column_stack([a,f(w*b)]); r=v-d@fit(d,v); return np.mean(r*r)**.5
    ws=np.linspace(.2,6,100); j=np.argmin([err(w) for w in ws]); res=minimize_scalar(err,bounds=(ws[max(0,j-1)],ws[min(j+1,len(ws)-1)]),method='bounded'); print(f.__name__,round(res.x,5),round(res.fun,7),end=' / ')
   print()
