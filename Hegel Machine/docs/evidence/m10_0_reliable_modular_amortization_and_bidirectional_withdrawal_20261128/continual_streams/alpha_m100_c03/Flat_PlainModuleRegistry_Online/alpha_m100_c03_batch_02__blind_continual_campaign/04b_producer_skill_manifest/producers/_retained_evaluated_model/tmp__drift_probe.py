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
from scipy.optimize import least_squares
xx=m._features(q,q)[len(h):]; drift=th[2]+xx@th[3:]; mech=((h[:,None]-z)*th[:2])@b[:,:2].T
f=np.column_stack((np.ones(len(h)),drift,h,z)); res=y-mech
print('diag drift range',np.quantile(drift,[0,.25,.5,.75,1])); print('joint transfer rows',np.linalg.lstsq(f,res,rcond=None)[0].round(5)); print('diag features missing',np.isnan(q['coupon_features_raw']).sum(0));
laws={'saturation':lambda a: -np.expm1(-t/a[0])*a[0], 'rational':lambda a:t/(1+t/a[0]), 'cutoff':lambda a:t*np.exp(-t/a[0]), 'tanh':lambda a:a[0]*np.tanh(t/a[0]), 'nonlinear rational':lambda a:t/(1+(t/a[0])**a[1]), 'soft cutoff':lambda a:t*np.exp(-(t/a[0])**a[1]), 'saturation+linear':lambda a:a[1]*(-np.expm1(-t/a[0])*a[0])+(1-a[1])*t}
base=np.r_[t,1]; labels=q['lab_id'];
def metrics(e):
 return np.round([np.sqrt(np.mean(e[:,:-1]**2)),max(np.sqrt(np.mean(e[labels==l,:-1]**2)) for l in np.unique(labels)),max(np.sqrt(np.mean(e[:,j]**2)) for j in range(len(t))),np.sqrt(np.mean(e[:,-1]**2)),max(np.sqrt(np.mean(e[labels==l,-1]**2)) for l in np.unique(labels))],5)
for name,fun in laws.items():
 e=res.copy(); pars=[]
 for lab in list(np.unique(labels))+[None]:
  keep=labels!=lab if lab is not None else np.ones(len(labels),bool)
  mod=lambda p: drift[:,None]*np.r_[fun(p),1]
  a=least_squares(lambda p:(mod(p)-res)[keep].ravel(),[1.5,1.] if name in ('nonlinear rational','soft cutoff','saturation+linear') else [1.5],bounds=(.025,20),loss='soft_l1',f_scale=.006).x
  if lab is not None: e[~keep]=(res-mod(a))[~keep]
  else: print(name,a.round(5),'LOLO',metrics(e),'fit',metrics(res-mod(a)))
for name,bs,rr in [('unconstrained',np.eye(len(t)+1),res),('same modes',b,res),('unit derivative',np.column_stack((b[:,:2]-np.outer(base,1/tau),)),res-drift[:,None]*base)]:
 e=res.copy()
 for lab in np.unique(labels):
  mask=labels!=lab; v=drift[mask]@rr[mask]/(drift[mask]@drift[mask]); e[~mask]=rr[~mask]-drift[~mask,None]*(bs@np.linalg.lstsq(bs,v,rcond=None)[0]);
 v=drift@rr/(drift@drift); print(name,'drift response',(bs@np.linalg.lstsq(bs,v,rcond=None)[0]+(base if name=='unit derivative' else 0)).round(5),'LOLO',metrics(e));
