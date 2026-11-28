import sys, pickle, importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
import _verified_upstream as u
spec=importlib.util.spec_from_file_location('arch','/tmp/hegel_continual_archive/alpha_m100_c01_batch_02_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=u.calibrate_upstream(dict(p['dataset'],chamber_verification_trials=p['campaign_diagnostics'])); tr,te,di=[d[k] for k in ('train','test','chamber_verification_trials')]; t,dt=d['eval_times'],d['past_dt']
xtr,xte=a._features(tr,te)
def inputs(s,x): return s['past_channel_raw'].sum(axis=-1),s['hold_channel_raw'].sum(axis=-1),np.column_stack((np.ones(len(x)),x))
def obs(s):return s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
tau,b=a._fit(*inputs(tr,xtr),dt,t,*obs(tr));print('taus',tau,'theta',b)
c,r=a._design(tau,*inputs(tr,xtr),dt,t); yc,yr=obs(tr); print('train abs error quantiles',np.quantile(abs(c@b-yc),[.5,.9,.95,.99,1]),np.quantile(abs(r@b-yr),[.5,.9,.95,.99,1]))
ref= {k:np.concatenate([s[k] for s in (tr,te)]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; _,xd=a._features(ref,di)
# Also apply the exact train/test standardization and observed-cell imputation to diagnostics without adding their rows to the panel.
x0=di['coupon_features_raw'].copy(); xref=ref['coupon_features_raw'].copy()
for l in np.unique(ref['lab_id']):
 mask=ref['lab_id']==l; R=ref['ref_coupons'][mask]; m=np.median(R.mean(0)); s=np.median(R.std(0)); xref[mask]=(xref[mask]-m)/s; msk=di['lab_id']==l; x0[msk]=(x0[msk]-m)/s
_,i,n=np.unique(np.column_stack((ref['lab_id'],ref['stratum'])),axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]; x0=np.where(np.isnan(x0),np.nansum(xref*w,0)/np.sum(w*~np.isnan(xref),0),x0)
past,h,F=inputs(di,x0); Y,R=obs(di); age=np.cumsum(dt[::-1])[::-1]-dt; Z=past@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); G=h[:,None]-Z; phi=-np.expm1(-t[:,None]/tau); Q=np.vstack((np.column_stack((phi,t)),np.r_[1/tau,1])); data=np.column_stack((Y,R)); amp=np.linalg.lstsq(Q,data.T,rcond=None)[0].T
print('free amplitudes residual quantiles',np.quantile(abs(amp@Q.T-data),[.5,.9,.99,1])); print('feature convention difference',np.quantile(abs((x0-xd)@b[3:]),[.5,.9,1])); print('corr amp vs hold, memories, first/last, drift',np.round(np.corrcoef(np.column_stack((amp,h,Z,past[:,[0,-1]],F@b[2:])).T)[:3,3:],3)); print('amp snapshot [h,z0,z1,d,A0,A1,D]', np.round(np.column_stack((h,Z,F@b[2:],amp))[:12],5)); print('fitted coefficients amp vs hold,z,drift,intercept',np.linalg.lstsq(np.column_stack((h,Z,F@b[2:],np.ones(len(h)))),amp,rcond=None)[0]);
for name,gb in [('baseline',G),('reset',np.broadcast_to(h[:,None],G.shape))]:
 L=np.column_stack((gb*b[:2],F@b[2:])); err=L@Q.T-data; print(name,'curve/rate med abs',np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])))
