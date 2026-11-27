import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,g,t,Y,R,theta=p.m,p.g,p.t,p.Y,p.R,p.theta
x=m._features(g,g)[0]; past,h,F=m._specimen(g,x,p.bridge); q=F@theta[2:]; E=Y-(h[:,None]*theta[:2])@R.T
T=np.r_[t,1.]; labs=g['lab_id']; n=len(q)
cal=p.d['calibration_load_cells']; raw=cal['bridge_raw']; y=cal['true_channel_loads'].sum(1); e=(raw-p.bridge[0])@p.bridge[1]-y
print('bridge total residual quantiles',np.quantile(e,[0,.1,.5,.9,1]),'load range',np.min(cal['true_channel_loads'],0),np.max(cal['true_channel_loads'],0)); print('sweep keys',list(cal['single_channel_sweeps']))
c,b=m._branches(p.sp,p.dt,t,p.tau); qt=p.sp[2]@theta[2:]; ytr=p.observed(p.tr); err=np.c_[c@theta[:2]+qt[:,None]*t,b@theta[:2]+qt]-ytr
print('train residual quantiles curve/rate',np.quantile(err[:,:-1],[.05,.25,.5,.75,.95]),np.quantile(err[:,-1],[.05,.25,.5,.75,.95])); print('train rate residual correlations to z, hold, features, aux',np.round(np.corrcoef(np.c_[err[:,-1],p.sp[0],p.sp[1],p.xt,p.tr['aux_chamber_telemetry']].T)[0,1:],3))
K=np.r_[np.log1p(t),1.]; a=np.linalg.lstsq(np.stack([T,K],axis=1),E.T,rcond=None)[0].T
print('mode amplitude partial regressions:')
for key,A in [('q',np.c_[q,q*0+1]),('q+h',np.c_[q,h,q*0+1]),('q+q²+h',np.c_[q,q*q,h,q*0+1]),('features+h',np.c_[F,h]),('q+h+past',np.c_[q,h,past,q*0+1])]:
 co=np.linalg.lstsq(A,a,rcond=None)[0]; res=a-A@co; print(key,np.round(co.T,3).tolist(),'rmse',np.round(np.sqrt(np.mean(res**2,axis=0)),4))
print('row residuals by lab with quadratic model:')
A=np.c_[q,q*q,h,q*0+1]; res=a-A@np.linalg.lstsq(A,a,rcond=None)[0]
print(np.round(np.c_[labs,g['stratum'],np.isnan(g['coupon_features_raw']).any(1),q,h,a,res],3))
# Compare shared differentiable time bases under the same out-of-lab tests.
for label,base in [('reset',(h[:,None]*theta[:2])@R.T),('bench',Y*0)]:
 if label=='bench':
  c,b=m._branches((past,h,F),p.dt,t,p.tau); base=np.c_[c@theta[:2],b@theta[:2]]
 for tag,K in [('exp',np.r_[-np.expm1(-t),1.]),('log',np.r_[np.log1p(t),1.]),('rat',np.r_[t/(1+.6*t),1.]),('sqrt',np.r_[t/np.sqrt(1+t),1.])]:
  for nm,A in [('linear',np.c_[q,h,q*0+1]),('quadratic',np.c_[q,q*q,h,q*0+1]),('quadratic+h*q',np.c_[q,q*q,h,(h-2.3)*q,q*0+1])]:
   D=np.einsum('ni,bk->nkib',A,[T,K]).reshape(n,len(T),-1); pred=Y.copy()
   for lab in np.unique(labs):
    ok=labs!=lab; b=np.linalg.lstsq(D[ok].reshape(-1,D.shape[-1]),(Y-base)[ok].ravel(),rcond=None)[0]; pred[~ok]=base[~ok]+D[~ok]@b
   err=pred-Y; score=np.array([[np.sqrt(np.mean(err[j,:-1]**2)),np.sqrt(np.mean(err[j,-1]**2))] for j in [labs==l for l in np.unique(labs)]])
   print(label,tag,nm,'worst',np.round(score.max(0),4),'pooled',np.round([np.sqrt(np.mean(err[:,:-1]**2)),np.sqrt(np.mean(err[:,-1]**2))],4))
