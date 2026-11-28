import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf,expit
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_b'][k] for k in ('u','v')); n,d=u.shape
funcs={'sine':np.sin,'tanh':np.tanh,'atan':np.arctan,'erf':erf,'softsign':lambda z:z/(1+np.abs(z)),'asinh':np.arcsinh,'rat':lambda z:z/(1+z*z),'sqrt':lambda z:z/np.sqrt(1+z*z),'cube':lambda z:z**3,'logistic_deriv':lambda z:expit(z)*expit(-z),'relu':lambda z:np.maximum(z,0)}
rng=np.random.default_rng(971)
for name,f in funcs.items():
 for linear in (0,1):
  def design(p,x):
   g=f(x*np.exp(p[:4])+p[4:]); return np.c_[np.ones(len(x)),x,g] if linear else np.c_[np.ones(len(x)),g]
  def solve(p,tr):
   A=design(p,u[tr]); return np.linalg.lstsq(A,v[tr],rcond=None)[0]
  def fun(p,tr):return (design(p,u[tr])@solve(p,tr)-v[tr]).ravel()
  tr=np.arange(n); lo=np.r_[[-2.]*4,[-8.]*4];hi=np.r_[[2.7]*4,[8.]*4]; starts=[np.r_[[np.log(k)]*4,[0.]*4] for k in [0.5,1,2,3,4,5,6,8,11]]+[np.r_[rng.uniform(0,2.1,4),rng.uniform(-2.,2.,4)] for _ in range(12)]
  best=min((least_squares(fun,p,args=(tr,),bounds=(lo,hi),max_nfev=90) for p in starts),key=lambda o:o.cost)
  rms=(2*best.cost/v.size)**.5
  if rms<.045:
   errs=[]
   for k in range(4):
    te=tr[tr%4==k]; tt=tr[tr%4!=k]; fit=min((least_squares(fun,p,args=(tt,),bounds=(lo,hi),max_nfev=90) for p in starts),key=lambda o:o.cost); errs.extend((design(fit.x,u[te])@solve(fit.x,tt)-v[te]).ravel())
   print(name,linear,rms,'fold MAE/max',np.mean(np.abs(errs)),np.max(np.abs(errs)),'scale',np.exp(best.x[:4]),'shift',best.x[4:],flush=True)
   if rms<.005:print(solve(best.x,tr),flush=True)
  else: print(name,linear,rms,flush=True)
