import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/probe_chamber.py')
m=p['m']; tr,te,q,cal,dt,t=(p[k] for k in ('tr','te','q','cal','dt','t')); taus,theta=p['taus'],p['theta']; Y=np.c_[p['y'],p['yr']]; w=np.r_[np.ones(len(t)),.8]; labs=q['lab_id']; strata=q['stratum'];
A=lambda u: np.r_[-np.expm1(-t/u),1/u]
X=m._features(q,q)[:len(labs)]; B,C=m._design(taus,m._inputs(q,X,cal),dt,t); G=np.c_[B[:,:,:2]@theta[:2],C[:,:2]@theta[:2]]; drift=theta[2]+X@theta[3:]; hold=m._inputs(q,X,cal)[1]
for label,a,b in [('train',tr,te),('train-only',tr,tr),('test-only',te,te),('diag-only',q,q)]:
 x=m._features(a,b)[:len(a['lab_id'])]; print(label,'means/sd of drift',np.mean(theta[2]+x@theta[3:]),np.std(x@theta[3:]))
 if a is tr:
  aa,bb=m._design(taus,m._inputs(a,x,cal),dt,t); print('train MAE with fixed parameters',np.median(abs(aa@theta-(a['strain_curves_obs']-a['strain_0'][:,None]))))
print('diag drift, observed effective initial drift:',np.c_[drift,Y[:,-1]-G[:,-1]].round(4).tolist())
L=np.r_[t,1.]; slow=np.c_[B[:,:,1]*theta[1],C[:,1]*theta[1]]; fast=G-slow
for kind in ('fixed-unit','unit','amp','amp-linear','shift-slow','amp-log','amp-affine','amp-gaps'):
 def pred(v):
  if kind=='fixed-unit': return G+drift[:,None]*A(1.)
  u=v[0]; h=u*A(u); g=G
  if kind=='unit': d=drift
  elif kind=='shift-slow':
   g=fast+slow[:,-1,None]*v[1]*A(v[1]); d=drift*v[2]
  elif kind=='amp-log': d=v[1]*drift+v[2]*drift**2
  elif kind=='amp-affine': d=v[1]*drift+v[2]
  elif kind=='amp-gaps': d=v[1]*drift+v[2]*fast[:,-1]+v[3]*slow[:,-1]
  else: d=v[1]*drift
  out=g+d[:,None]*h
  if kind=='amp-linear': out+=v[2]*drift[:,None]*L
  return out
 n={'unit':1,'amp':2,'amp-linear':3,'shift-slow':3,'amp-log':3,'amp-affine':3,'amp-gaps':4}.get(kind,0)
 v0=np.array([1.,4.17,1.] if kind=='shift-slow' else [1.,1.,0.,0.][:n]); lb=np.r_[.15, .75 if kind=='shift-slow' else -4.,-4.,-4.][:n]; ub=np.r_[6.,12. if kind=='shift-slow' else 4.,4.,4.][:n]
 def fit(m):
  if n==0:return v0
  return least_squares(lambda z:((pred(z)-Y)*w)[m].ravel(),v0,bounds=(lb,ub),loss='soft_l1',f_scale=.005).x
 v=fit(np.ones(len(X),bool)); ms=[]
 for groups in (np.arange(len(labs)),labs,strata):
  e=np.zeros_like(Y)
  for j in np.unique(groups):
   s=groups==j;e[s]=(pred(fit(~s))-Y)[s]
  ms.append([round(np.sqrt(np.mean(e**2,0)).max(),5),round(np.sqrt(np.mean(e[:,-1]**2)),5),round(max(np.sqrt(np.mean(e[groups==j]**2,0)).max() for j in np.unique(groups)),5)])
 print(kind,v.round(5),'LOO, LOLO, LOSO [worst horizon RMSE, rate RMSE, worst group/horizon RMSE]',ms)
