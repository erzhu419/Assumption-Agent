import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp'); import prior as p
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; g=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; s=p._stress_map(d['calibration_load_cells']); tau,th=p._fit(tr,p._features(tr,te)[0],s,dt,t)
x=p._features(tr,g)[1]; xc=p._features(dict((k,np.concatenate([tr[k],te[k]])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')),g)[1]; h=s[0]+g['hold_channel_raw']@s[1:]; past=s[0]+g['past_channel_raw']@s[1:]; drift=th[2]+x@th[3:]; driftc=th[2]+xc@th[3:]; one=np.ones(len(h)); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; labs=g['lab_id']
def shape(u): return np.r_[np.c_[-np.expm1(-t[:,None]/u),t], [np.r_[1/u,1.]]]
def prof(q):
 A=shape(np.exp(q)); return (np.linalg.lstsq(A,Y.T,rcond=None)[0].T@A.T-Y).ravel()
u=np.exp(least_squares(prof,np.log(tau)).x); print('chamber',u,'material',tau)
for u0 in [tau,u]:
 A=shape(u0); V=np.linalg.lstsq(A,Y.T,rcond=None)[0].T; print('amplitudes at',u0,'singular values',np.linalg.svd(A,compute_uv=False))
 age=np.r_[np.cumsum(dt[::-1])[::-1][1:], 0.]; z=past@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); zd=past@(-np.expm1(-dt[:,None]/u0)*np.exp(-age[:,None]/u0))
 for name, X in [('hold',np.c_[one,h]),('hold+drift',np.c_[one,h,drift]),('hold+driftc',np.c_[one,h,driftc]),('hold+slow+drift',np.c_[one,h,z[:,1],drift]),('hold+both+drift',np.c_[one,h,z,drift]),('hold+both',np.c_[one,h,z]),('chamber history',np.c_[one,h,zd,drift]),('hold+features',np.c_[one,h,x]),('hold+history+features',np.c_[one,h,past,x]),('drift smooth',np.c_[one,h,drift,np.maximum(drift,0),drift**2,drift**3]),('hold+features central',np.c_[one,h,xc]),('past6',np.c_[one,h,past])]:
  B=np.linalg.lstsq(X,V,rcond=None)[0]; pred=np.empty_like(V)
  for lab in np.unique(labs):
   i=labs!=lab; pred[~i]=X[~i]@np.linalg.lstsq(X[i],V[i],rcond=None)[0]
  err=pred@A.T-Y
  print(name,'coef',np.round(B,3).tolist(),'fit amp',np.round(np.mean((X@B-V)**2,axis=0)**.5,4),'lolo amp',np.round(np.mean((pred-V)**2,axis=0)**.5,4),'lolo y/r',np.round([np.mean(err[:,:-1]**2)**.5,np.mean(err[:,-1]**2)**.5],4))
 print('fast offset from th*h / drift', np.round(np.c_[V[:,0]-th[0]*h,drift,V[:,1]-V[:,2]*25],3))
print('sweep'); c=d['calibration_load_cells']['single_channel_sweeps'];print(np.round(np.column_stack(list(c.values())),4))
