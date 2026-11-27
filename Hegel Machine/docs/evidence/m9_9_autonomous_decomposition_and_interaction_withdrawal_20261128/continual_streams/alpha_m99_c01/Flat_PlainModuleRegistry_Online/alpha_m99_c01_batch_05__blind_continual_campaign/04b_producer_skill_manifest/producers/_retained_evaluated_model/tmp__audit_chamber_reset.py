import sys,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; out=solve_batch(d); tau=out['estimated_taus']; beta=out['hereditary_weights']; t=d['eval_times']; dt=d['past_dt']
c=calibrate_upstream(dict(d,chamber_verification_trials=item['campaign_diagnostics']))['chamber_verification_trials']; labs=c['lab_id']; strata=c['stratum']; x=c['coupon_features_raw'].copy(); complete=np.isfinite(x).all(1)
# The verification panel has its own reference coordinates and mean fill.
for l in np.unique(labs):
 s=labs==l; ref=c['ref_coupons'][s]; x[s]=(x[s]-np.median(ref.mean(0)))/np.median(ref.std(0))
x=np.where(np.isfinite(x),x,np.nanmean(x,axis=0)); drift=beta[2]+x@beta[3:]; hold=c['hold_channel_raw'].sum(-1); p=c['past_channel_raw'].sum(-1)
z=p@(np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); jump=hold-p[:,-1]
B=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.])).T
obs=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; base=(hold[:,None]*beta[:2])@B[:2]; y=obs-base
mode=lambda f,b:f[:,None]*B[b]
core=[mode(drift,j) for j in range(3)]
challengers={'reset':[], 'retain history':[mode(z[:,j],j) for j in range(2)], 'last-step difference':[mode(jump,j) for j in range(2)], 'rational history':[mode(z[:,j]/(1+z[:,j]**2),j) for j in range(2)], 'hold/history/drift coupling':[mode(hold*z[:,j]*drift,j) for j in range(2)], 'stratum-dependent drift':[mode(drift*(strata==k),2) for k in np.unique(strata)[1:]]}
rms=lambda v: np.sqrt(np.mean(v**2))
for name,extra in challengers.items():
 A=np.stack(core+extra,axis=-1); e=np.empty_like(y)
 for l in np.unique(labs):
  s=labs==l; fit=~s&complete; coef=np.linalg.lstsq(A[fit].reshape(-1,A.shape[-1]),y[fit].ravel(),rcond=None)[0]; e[s]=A[s]@coef-y[s]
 print(name,'pooled strain/rate',round(rms(e[:,:-1]),6),round(rms(e[:,-1]),6),'worst lab-horizon / lab rate',round(max(np.sqrt(np.mean(e[labs==l,:-1]**2,axis=0)).max() for l in np.unique(labs)),6),round(max(rms(e[labs==l,-1]) for l in np.unique(labs)),6))
 if name=='reset':
  for l in np.unique(labs):
   s=labs==l; print('  lab',l,'signed strain/rate bias',np.round([e[s,:-1].mean(),e[s,-1].mean()],6),'strain/rate RMSE',np.round([rms(e[s,:-1]),rms(e[s,-1])],6))
