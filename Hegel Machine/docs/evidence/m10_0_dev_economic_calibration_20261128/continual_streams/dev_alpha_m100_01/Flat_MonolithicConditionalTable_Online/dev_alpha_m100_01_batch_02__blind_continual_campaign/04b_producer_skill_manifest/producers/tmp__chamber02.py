import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import timefit02 as u
from explore02 import a,b,c,t,dt,br,pc,hc,tau,th,Y,rate,rms
from material02 import _features,_design
import numpy as np
from scipy.optimize import minimize_scalar
n=len(hc); K=len(t)
# Remove only the hereditary benchtop component, then model the entire chamber drift.
C,R=_design(tau,pc,hc,np.zeros((n,4)),dt,t)
obs=np.c_[Y-C[:,:,:2]@th[:2],rate-R[:,:2]@th[:2]]
nd=np.isnan(c['coupon_features_raw']).any(axis=1)
for label,z in [('abc',u.xc),('ab',u.xx),('diag',_features(c,c)[:n])]:
 dr=th[2]+z@th[3:]; h=hc-hc.mean(); m=np.ones(n)
 for label2,X in [('intrinsic',dr[:,None]),('offset',np.c_[dr,m]),('hold',np.c_[dr,m,h]),('features',np.c_[m,z]),('features+hold',np.c_[m,z,h])]:
  def fit(idx):
   def solve(logk,full=False):
    k=np.exp(logk); basis=np.r_[-np.expm1(-t/k),1/k]; vv=X[idx]; yy=obs[idx]; beta=np.linalg.lstsq(vv,yy@basis/(basis@basis),rcond=None)[0]; e=vv@beta[:,None]*basis-yy
    return (k,beta,basis) if full else np.mean(e*e)
   out=minimize_scalar(solve,bounds=(-1.4,1.4),method='bounded'); return solve(out.x,True)
  k,coef,bb=fit(np.arange(n)); e=np.array([X[i]@bet*basis-obs[i] for i in range(n) for _,bet,basis in [fit(np.flatnonzero(np.arange(n)!=i))]]); el=np.concatenate([X[idx]@bet[:,None]*basis-obs[idx] for lab in np.unique(c['lab_id']) for idx in [c['lab_id']==lab] for _,bet,basis in [fit(~idx)] ]); print(label,label2,'k',round(k,5),'coef',np.round(coef,5),'fit',np.round(rms(X@coef[:,None]*bb-obs)[[0,-2,-1]],5),'LOO worst-time/rate',np.round([max(rms(e)[:-1]),rms(e)[-1]],5),'leave-lab',np.round([max(rms(el)[:-1]),rms(el)[-1]],5),'missing max',round(abs(e[nd]).max(),4));
 print('matched pole amplitudes at k=1: hold/dr/rms',np.round(np.linalg.lstsq(np.c_[dr,m,h],obs@np.r_[-np.expm1(-t),1]/(np.r_[-np.expm1(-t),1]@np.r_[-np.expm1(-t),1]),rcond=None)[0],5))
print('train fit spectrum',tau,th)
