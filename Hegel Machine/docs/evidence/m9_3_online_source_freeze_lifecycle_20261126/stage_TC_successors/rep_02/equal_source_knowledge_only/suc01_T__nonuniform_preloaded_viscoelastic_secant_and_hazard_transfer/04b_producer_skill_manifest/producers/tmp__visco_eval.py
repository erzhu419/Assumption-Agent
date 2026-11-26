import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import preloaded_hereditary_secant_hazard_solver as m
for item in pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb')):
 d,g = item['dataset'],item['gt']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; x,xx=m._panel(tr,te)
 for mode in [False,True]:
  tau,c,cost=m._fit(tr,dt,t,x,mode); a,b=m._design(te,dt,t,tau); aa,bb=m._design(tr,dt,t,tau)
  print('fit',mode,tau,c,'cost',cost)
  rd=tr['instant_rate_obs']-bb@c[:2]
  drift0=xx@np.linalg.lstsq(x,rd,rcond=None)[0] if mode else xx@c[2:]
  re=te['instant_rate_obs']-b@c[:2]; init=te['strain_0']; raw=tr['strain_curves_obs']-tr['strain_0'][:,None]
  print('training curve residual quantiles',np.quantile(np.abs(raw-aa@c[:2]-rd[:,None]*t),[.25,.5,.75,.90,.95]),'rate prior quantiles',np.quantile(np.abs(rd-x@np.linalg.lstsq(x,rd,rcond=None)[0]),[.25,.5,.75,.9,.95]))
  for k in [0,.25,.5,.75,1]:
   curve=init[:,None]+a@c[:2]+((1-k)*drift0+k*re)[:,None]*t
   sec=(curve[:,-1]-init)/np.maximum(np.abs(te['stress_hold_0']),.5)
   cross=m._crossings(init,curve,t,te['critical_strain'])
   err=np.column_stack([np.abs(curve-g['te_clean']).mean(1),np.abs(sec-g['te_secant']),np.abs(cross-g['te_cross'])]); lab=te['lab_id']; miss=~np.isfinite(te['coupon_features_raw']).all(1)
   print('weight',k,'pooled',err.mean(0),'worst lab',np.max([err[lab==v].mean(0) for v in np.unique(lab)],axis=0),'missing slice',err[miss].mean(0))
