import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/compare_maps.py').read())
from scipy.optimize import minimize_scalar
U,V=b['u'],b['v']; A=features(U,1); u=base['deA']; print('archived');print(open('/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py').read()); print('frozen');print(open('/holdout_task/_frozen_nodes/node_02.py').read())
lib={'tanh':lambda u,k:np.tanh(k*u),'sin':lambda u,k:np.sin(k*u),'rational':lambda u,k:u/(1+k*u*u),'arcsinh':lambda u,k:np.arcsinh(k*u),'atan':lambda u,k:np.arctan(k*u),'cubic':lambda u,k:u**3,'softsign':lambda u,k:u/(1+k*abs(u))}
for name,f in lib.items():
 def fit(k,ix=np.arange(len(U)),linear=True):
  def F(u): return np.c_[features(u,1) if linear else np.ones(len(u)),f(u,k)]
  cf=np.linalg.lstsq(F(U[ix]),V[ix],rcond=None)[0]
  return lambda z:F(z)@cf,cf
 for linear in [True,False]:
  def learn(ix):
   def obj(k):return np.mean((fit(k,ix,linear)[0](U[ix])-V[ix])**2)
   ks=np.linspace(.025,6,55);j=np.argmin([obj(k) for k in ks]);k=minimize_scalar(obj,bounds=(ks[max(0,j-1)],ks[min(j+1,len(ks)-1)]),method='bounded').x; return k,obj(k)
  ix=np.arange(len(U));k,m=learn(ix);fn,cf=fit(k,ix,linear);le=[np.linalg.norm(fit(learn(ix[ix!=i])[0],ix[ix!=i],linear)[0](U[i:i+1])[0]-V[i]) for i in ix]; print(name,linear,'k',round(k,5),'cal rmse',round(m**.5,6),'LOO mean/max',round(np.mean(le),6),round(max(le),6),'scales',[(sc,score(fn(u)*sc).round(5).tolist()) for sc in [.35,.6,1,1.5,2.5]]);print('coef',cf.round(3)) if m<.0001 else None
