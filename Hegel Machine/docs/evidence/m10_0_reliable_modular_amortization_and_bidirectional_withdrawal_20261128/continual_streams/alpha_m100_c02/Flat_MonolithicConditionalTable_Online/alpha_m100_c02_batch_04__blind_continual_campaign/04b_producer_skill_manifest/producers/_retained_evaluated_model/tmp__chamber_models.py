exec(open('/tmp/chamber_probe.py').read().split("for tag,s,h")[0])
from scipy.linalg import block_diag
Y=v['strain_curves_obs']-v['strain_0'][:,None]; R=v['instant_rate_obs']; n=len(R)
a,b,*_=m._design(np.log(tau),pv,hv,xv,dt,t); z=hv[:,None]-b[:,:2]*tau; drift=xv@theta[2:]; print('diag fields',[(k,getattr(q,'shape',None)) for k,q in v.items()]); print('test fields',list(te)); print('times',t)
T=np.column_stack([-np.expm1(-t[:,None]/tau),t]); T=np.vstack([T,np.r_[1/tau,1]]); target=np.c_[Y,R]; amp=np.linalg.lstsq(T,target.T,rcond=None)[0].T; print('amp head',amp[:8].round(5));
for tag,f in [('H',np.c_[np.ones(n),hv]),('Hz',np.c_[np.ones(n),hv,z]),('Hd',np.c_[np.ones(n),hv,drift]),('HdZ',np.c_[np.ones(n),hv,drift,z]),('Hdf',np.c_[np.ones(n),hv,drift,pv[:,-1]]),('HXd',np.c_[hv,xv]),('HZXd',np.c_[hv,z,xv]),('Hdall',np.c_[np.ones(n),hv,drift,pv])]:
 for offset in [np.zeros_like(amp),np.c_[hv[:,None]*theta[:2],drift]] if tag in ['Hd','HdZ','Hdf'] else [np.zeros_like(amp)]:
  off=offset@T.T
  fit=lambda k:np.linalg.lstsq(f[k],(target-off)[k],rcond=None)[0]@np.linalg.pinv(T.T)
  pred=np.empty_like(target)
  for lab in np.unique(v['lab_id']):
   idx=v['lab_id']==lab; pred[idx]=off[idx]+f[idx]@fit(~idx)@T.T
  err=abs(pred-target); print(tag,'offset',bool(offset.any()),'median/max',np.median(err[:,:-1]).round(5),np.max(err[:,:-1]).round(5),np.median(err[:,-1]).round(5),np.max(err[:,-1]).round(5),'coef',fit(slice(None)).round(4).tolist())
print('features joint vs diag local vs fixed ref')
for other in [tr,te,dict((k,np.concatenate([tr[k],te[k]])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw'))]:
 xx=m._features(other,v)[-n:]; print(np.max(abs(xx-xv),axis=0),np.std((xx-xv)@theta[2:]))
