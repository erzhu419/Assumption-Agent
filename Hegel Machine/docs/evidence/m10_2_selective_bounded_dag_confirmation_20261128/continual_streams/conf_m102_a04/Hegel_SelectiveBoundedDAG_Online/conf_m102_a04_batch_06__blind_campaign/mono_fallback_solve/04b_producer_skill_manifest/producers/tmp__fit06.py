import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=6,suppress=True,linewidth=160)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(a): return np.column_stack((np.ones(len(a)),a))
def poly(x,d): return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def align(p):
 u=one(p['u']); v=p['v']; rng=np.random.default_rng(721); best=None
 for _ in range(1000):
  i=rng.choice(len(u),5,False); b=ls(u[i],v[i]); r=np.linalg.norm(u@b-v,axis=1); score=np.sort(r)[:34].mean()
  if best is None or score<best[0]: best=(score,b)
 b=best[1]
 for _ in range(5):
  r=np.linalg.norm(u@b-v,axis=1); keep=r<max(4*np.median(r),.001); b=ls(u[keep],v[keep])
 return b,r
p=s['panel_a']; a,r=align(p); print('a',a,'sorted r',np.sort(r));
u=s['panel_c']['u'];
for k in ['v1','v2']:
 v=s['panel_c'][k]; print(k)
 def ff(kind,q):
  if kind=='softsign': return one(u/(1+q[0]*abs(u))+q[1]*u)
  if kind=='sin': return one(np.sin(q[0]*u)+q[1]*u)
  if kind=='tanh': return one(np.tanh(q[0]*u)+q[1]*u)
  if kind=='curv': return np.c_[one(u),u*u,u*u/(1+q[0]*u*u)]
  if kind=='poly2': return poly(u,2)
  if kind=='poly3sep': return np.c_[one(u),u*u,u**3,u**4]
  if kind=='interactions': return np.c_[poly(u,2),u**3,u**4]
 for kind in ('softsign','sin','tanh','curv','poly2','poly3sep','interactions'):
  def fun(q):
   f=ff(kind,q); return (f@ls(f,v)-v).ravel()
  if kind in ('poly2','poly3sep','interactions'): q=[1,0]
  else: q=least_squares(fun,[.9,0],bounds=([.01,-5],[10,5]),gtol=1e-12).x
  f=ff(kind,q); print(kind, q,'mae',abs(fun(q)).mean(),'52-like',abs(f[-6:]@ls(f[:-6],v[:-6])-v[-6:]).mean())
x=s['train']['x']; z=(x-a[0])@np.linalg.inv(a[1:]); b=ls(one(s['panel_b']['u']),s['panel_b']['v']); w=one(z)@b; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; print('z bounds',z.min(0),z.max(0),'w bounds',w.min(0),w.max(0)); print('b',b)
for nm,g in [('exp',lambda r:-np.expm1(-t[:,None]*r)),('exp+linear',lambda r:np.c_[t,-np.expm1(-t[:,None]*r)]),('exp+sin',lambda r:np.c_[-np.expm1(-t[:,None]*r[:2]),np.sin(t*r[2])])]:
 def fun(q):
  f=g(q); return (f@ls(f,tr[:52].T)-tr[:52].T).ravel()
 for q in ([.3,1.1,2.],) if nm!='exp+linear' else ([.4,1.8],):
  fit=least_squares(fun,q,bounds=(.001,10),gtol=1e-12); f=g(fit.x); print(nm,fit.x,'fit',abs(fun(fit.x)).mean(),'validation basis',abs(f@ls(f,tr[52:].T)-tr[52:].T).mean())
np.savez('/tmp/data06.npz',a=a,b=b,z=z,w=w)
