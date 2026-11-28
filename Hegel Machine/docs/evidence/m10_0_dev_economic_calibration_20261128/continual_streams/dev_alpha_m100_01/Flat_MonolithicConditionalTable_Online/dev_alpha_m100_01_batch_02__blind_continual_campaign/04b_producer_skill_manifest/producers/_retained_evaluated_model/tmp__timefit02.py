import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from explore02 import *
from material02 import _design
from scipy.optimize import least_squares
keys=('lab_id','stratum','ref_coupons','coupon_features_raw'); ab={k:np.concatenate([a[k],b[k]]) for k in keys}
for label,s in [('ab',ab),('c',c),('abc',{k:np.concatenate([ab[k],c[k]]) for k in keys})]:
 print(label,[(i, np.median(s['ref_coupons'][s['lab_id']==i].mean(0)),np.median(s['ref_coupons'][s['lab_id']==i].std(0))) for i in np.unique(s['lab_id'])]);
xx=c['coupon_features_raw'].copy(); labs=c['lab_id']; ll=ab['lab_id'];
for j in np.unique(ll):
 ref=ab['ref_coupons'][ll==j]; xx[labs==j]=(xx[labs==j]-np.median(ref.mean(0)))/np.median(ref.std(0))
_,inv,count=np.unique(np.c_[ll,ab['stratum']],axis=0,return_counts=True,return_inverse=True); w=1/count[inv]; std=ab['coupon_features_raw'].copy()
for j in np.unique(ll):
 ref=ab['ref_coupons'][ll==j]; std[ll==j]=(std[ll==j]-np.median(ref.mean(0)))/np.median(ref.std(0))
for k in range(4):
 ok=np.isfinite(std[:,k]); xx[~np.isfinite(xx[:,k]),k]=np.average(std[ok,k],weights=w[ok]);
print('dr differences', np.c_[dr,th[2]+xx@th[3:],np.isnan(c['coupon_features_raw']).any(1)])
for label,z in [('combined',xc),('fixed-ab',xx)]:
 cc,rr=_design(tau,pc,hc,z,dt,t); yy=np.c_[Y-cc@th,rate-rr@th]; drift=th[2]+z@th[3:]; A=np.c_[np.ones(len(hc)),drift,hc]; coef=np.linalg.lstsq(A,yy,rcond=None)[0]; print(label,'time coefficients',coef); print('in-sample RMS',rms(yy-A@coef));
 for k in (0,1,2):
  def f(p):
   T=np.r_[t,0]; pred=p[0]*T+p[1]*(-np.expm1(-T/np.exp(p[2]))); pred[-1]=p[0]+p[1]/np.exp(p[2]); return pred
  fit=least_squares(lambda p:f(p)-coef[k],[-1,1,0]); print(k,'one pole',fit.x,np.exp(fit.x[2]),rms(f(fit.x)-coef[k]));
