import numpy as np,pickle,itertools,sys
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
def one(x):return np.column_stack((np.ones(len(x)),x))
def fit(F,v):
 rng=np.random.default_rng(12); best=(np.inf,None)
 for i in range(350):
  g=rng.choice(len(F),min(len(F),F.shape[1]+2),False); b=ls(F[g],v[g]); e=np.linalg.norm(v-F@b,axis=1); score=np.median(e)
  if score<best[0]:best=score,b
 _,b=best
 for _ in range(5):
  e=np.linalg.norm(v-F@b,axis=1); g=e<max(6*np.median(e),.001);b=ls(F[g],v[g])
 return b,np.linalg.norm(v-F@b,axis=1)
def pair(u):return np.array([u[:,a]*u[:,b] for a,b in itertools.combinations_with_replacement(range(4),2)]).T
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True,linewidth=150)
 u,v=s['panel_a'].values()
 for n,F in [('aff',one(u)),('diagquad',one(np.c_[u,u*u])),('poly2',one(np.c_[u,pair(u)])),('diagcube',one(np.c_[u,u*u,u**3]))]:
  b,e=fit(F,v); print(n,'median',np.median(e),'sorted',np.sort(e)); print(b)
 p=s['panel_c']; u=p['u']
 for vv in ['v1','v2']:
  v=p[vv];print(vv)
  for k,F in [('quadratic',one(np.c_[u,pair(u)])),('quadratic pure',one(pair(u)))]: print(k,np.sqrt(np.mean((v-F@ls(F,v))**2)))
  for typ,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('softsign',lambda x:x/(1+abs(x))),('atan',np.arctan),('exp',np.exp),('abs',np.abs),('rational',lambda x:1/(1+x*x))]:
   def f(w,P):return one(np.c_[w,fn(P[0]*w+P[1:])]);
   def err(P):
    F=f(u,P);return (F@ls(F,v)-v).ravel()
   best=min((least_squares(err,[a,*([p]*4)],max_nfev=300) for a in [.75,1.35,2.4,3.5] for p in [0,.4]),key=lambda m:m.cost)
   print(typ,best.x,np.sqrt(2*best.cost/v.size))
