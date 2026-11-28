import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
P='/tmp/hegel_continual_archive/alpha_m100_c04_batch_06_solver.py'
s=importlib.util.spec_from_file_location('prior',P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; z=p['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; n=len(tr['lab_id']); x=m._features(tr,te); b=m._calibrate(d['calibration_load_cells']); tau,theta=m._fit(m._inputs(tr,x[:n],b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
exec(open("/tmp/bridge07.py").read().split("for degree in range")[0]); cal=d["calibration_load_cells"]; C=np.array([fit(np.c_[np.ones(len(l)), l[:,j]], y[:,j]) for j in range(4)]); print("diagonal affine",C); m._stress=lambda raw,bridge: ((raw-C[:,0])/C[:,1]).sum(axis=-1); m._bridge_raw=lambda load,bridge: C[:,0]+load*C[:,1]; b=C; tau,theta=m._fit(m._inputs(tr,x[:n],b),dt,t,tr["strain_curves_obs"]-tr["strain_0"][:,None],tr["instant_rate_obs"])
print('bridge',b,'tau',tau,'theta',theta)
cal=d['calibration_load_cells']; err=m._bridge_raw(cal['true_channel_loads'],b)-cal['bridge_raw']; print('cal raw abs error quantiles',np.quantile(abs(err),[.5,.9,.95,.98,.99,1])); print('sweeps',[(k,type(v).__name__,getattr(v,'shape',None),list(v)[:6] if isinstance(v,dict) else None) for k,v in cal['single_channel_sweeps'].items()]);
h=m._stress(z['hold_channel_raw'],b); past=m._stress(z['past_channel_raw'],b); y=np.column_stack((z['strain_curves_obs']-z['strain_0'][:,None],z['instant_rate_obs']));
def basis(q):
 return np.vstack((np.column_stack((-np.expm1(-t[:,None]/q),t)),np.r_[1/q,1]))
def profile(q):
 A=basis(np.exp(q)); C=np.linalg.lstsq(A,y.T,rcond=None)[0].T; return C,(C@A.T-y)
for k in (1,2,3):
 for start in ([tau[:1]] if k==1 else [tau,tau*.5,tau*2] if k==2 else [np.r_[tau,4.],np.r_[.2,tau]]):
  q=least_squares(lambda q:profile(q)[1].ravel(),np.log(start),bounds=(np.log(.025),np.log(50)),loss='soft_l1',f_scale=.007,max_nfev=180).x
  C,r=profile(q); print('free',np.exp(q),'mae',np.mean(abs(r),axis=0).round(5),'max',np.max(abs(r)))
C,r=profile(np.log(tau)); print('intrinsic basis mae',np.mean(abs(r),axis=0).round(5),'max',np.max(abs(r))); print('first six coefficient rows',np.c_[h,C][:6].round(5));
mem=np.zeros((len(h),2))
for j,v in enumerate(dt):
 e=np.exp(-v/tau); mem=e*mem+(1-e)*past[:,j,None]
for variant in ('diag-only','train-test reference','train-diag','all'):
 if variant=='diag-only': xx=m._features(z,z)[len(h):]
 if variant=='train-diag': xx=m._features(tr,z)[n:]
 if variant=='all':
  a={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; xx=m._features(a,z)[len(a['lab_id']):]
 if variant=='train-test reference':
  labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack((tr['ref_coupons'],te['ref_coupons'])); _,idx,cnt=np.unique(np.c_[labs,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); w=(1/cnt[idx])[:,None]; raws=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); obs=np.isfinite(raws); fill=(x*w*obs).sum(0)/(w*obs).sum(0); xx=z['coupon_features_raw'].copy();
  for lab in np.unique(labs):
   r0=refs[labs==lab]; sel=z['lab_id']==lab; xx[sel]=(xx[sel]-np.median(r0.mean(0)))/np.median(r0.std(0))
  xx=np.where(np.isfinite(xx),xx,fill)
 drift=theta[2]+xx@theta[3:]; A=basis(tau)
 for nm,F in [('mechanism',np.c_[h,mem,drift]),('plus raw history',np.c_[h,past,drift]),('feature drift free',np.c_[h,mem,np.ones(len(h)),xx])]:
  B=np.linalg.lstsq(F,C,rcond=None)[0]; errs=[]
  for lab in np.unique(z['lab_id']):
   sel=z['lab_id']==lab; cf=np.linalg.lstsq(F[~sel],C[~sel],rcond=None)[0]; rs=(F[sel]@cf@A.T-y[sel]); errs.append(np.sqrt(np.mean(rs**2,axis=0)))
  r1=F@B@A.T-y; print(variant,nm,'fit rms curve/rate',np.sqrt(np.mean(r1[:,:-1]**2)),np.sqrt(np.mean(r1[:,-1]**2)),'worst lab/horizon',np.max(np.array(errs)[:,:-1]),np.max(np.array(errs)[:,-1]),'coef',B.round(4).tolist())
