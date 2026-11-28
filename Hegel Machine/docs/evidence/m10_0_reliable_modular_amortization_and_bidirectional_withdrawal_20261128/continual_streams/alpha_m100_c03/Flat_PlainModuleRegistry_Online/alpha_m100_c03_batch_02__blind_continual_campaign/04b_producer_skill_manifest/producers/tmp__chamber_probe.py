import sys,pickle,json,importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m100_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); q=calibrate_upstream(dict(d,test=i['campaign_diagnostics']),pay)['test']; d=calibrate_upstream(d,pay); tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; n=len(tr['lab_id']); loads=lambda s: (s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1)); x=m._features(tr,te); tau,th=m._fit(x[:n],*loads(tr),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
for s in (tr,te):
 a=s['aux_chamber_telemetry']; print('telemetry quartiles',np.quantile(a,[0,.1,.5,.9,1],axis=0).round(3)); print('labs',[(int(l),a[s['lab_id']==l].mean(0).round(3)) for l in np.unique(s['lab_id'])]);
p,h=loads(q); y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); z=np.zeros((len(h),2));
for j,s in enumerate(dt): z=np.exp(-s/tau)*z-np.expm1(-s/tau)*p[:,j,None]
b=np.vstack((np.column_stack((-np.expm1(-t[:,None]/tau),t)),np.r_[1/tau,1])); c=y@np.linalg.pinv(b.T); print('basis error',np.quantile(abs(c@b.T-y),[.5,.9,.99,1]),'cond',np.linalg.cond(b)); print('per-channel amplitude stats',np.quantile(c-h[:,None]*np.r_[th[:2],0],[0,.25,.5,.75,1],axis=0).round(4));
for name,xx in [('tr+q',m._features(tr,q)[n:]),('q',m._features(q,q)[len(q['lab_id']):]),('tr+te+q',m._features({k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')},q)[n+len(te['lab_id']):])]:
 drift=th[2]+xx@th[3:]; strain,rate=m._predict_chamber(tau,th,xx,h,t,q['strain_0']); old=np.column_stack((strain-q['strain_0'][:,None],rate)); a,b0=m._design(tau,xx,p,h,dt,t); base=np.column_stack((a@th,b0@th)); print(name,'reset/benchtop MAE',np.mean(abs(old-y),axis=0).round(4),np.mean(abs(base-y),axis=0).round(4));
 for kn,F in [('drift',np.column_stack((np.ones(len(h)),drift))),('linear',np.column_stack((np.ones(len(h)),drift,h,z))),('features',np.column_stack((np.ones(len(h)),xx,h,z)))]:
  target=y-(h[:,None]*th[:2])@b[:,:2].T; pred=np.empty_like(y)
  for lab in np.unique(q['lab_id']):
   mask=q['lab_id']==lab; coef=np.linalg.lstsq(F[~mask],target[~mask],rcond=None)[0]@np.linalg.pinv(b.T); pred[mask]=F[mask]@coef@b.T-target[mask];
  print(kn,'coef', (np.linalg.lstsq(F,target,rcond=None)[0]@np.linalg.pinv(b.T)).round(3),'CV',np.sqrt(np.mean(pred**2,axis=0)).round(4))
