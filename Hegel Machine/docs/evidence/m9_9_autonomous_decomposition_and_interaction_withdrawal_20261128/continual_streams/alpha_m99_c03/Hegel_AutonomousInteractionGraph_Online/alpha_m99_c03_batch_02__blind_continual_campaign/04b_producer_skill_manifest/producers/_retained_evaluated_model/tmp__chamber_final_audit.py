import sys
sys.path.insert(0,'/holdout_task')
import json,pickle,numpy as np
import downstream_solver as m
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
raw=item['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); dt=raw['past_dt']; t=raw['eval_times']; cal=raw['calibration_load_cells'];
fit=np.array([np.linalg.lstsq(np.c_[cal['true_channel_loads'][:,i],np.ones(len(cal['bridge_raw']))],cal['bridge_raw'][:,i],rcond=None)[0] for i in range(4)])
print('dynamic bridge minus payload, gain/offset',fit-np.array([pay['gains'],pay['offsets']]).T)
for label,payload in [('staged',pay),('dynamic',None)]:
 d=calibrate_upstream(raw,payload); diag=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),payload)['test']; sol=m.solve_downstream(d)
 tau,th=sol['estimated_taus'],sol['hereditary_weights']; print(label,'intrinsic',tau,th)
 x=m._features(diag); labs=diag['lab_id']; complete=np.isfinite(diag['coupon_features_raw']).all(1); strain,rate=m._predict(diag,x,tau,th,dt,t); e=strain-diag['strain_curves_obs']; er=rate-diag['instant_rate_obs'];
 for name,mask in [('all',np.ones(len(rate),bool)),('complete',complete),('missing',~complete)]+[(f'lab{l}',labs==l) for l in np.unique(labs)]:
  print(label,name,'strain/rate RMSE',np.mean(e[mask]**2)**.5,np.mean(er[mask]**2)**.5)
 if label=='staged':
  composed=solve_batch(raw)
  for k in sol: np.testing.assert_array_equal(composed[k],sol[k])
  eps=1.e-6; start,_=m._predict(diag,x,tau,th,dt,np.array([0.,eps])); np.testing.assert_allclose(start[:,0],diag['strain_0'],rtol=0,atol=1e-14); np.testing.assert_allclose((start[:,1]-start[:,0])/eps,rate,rtol=2e-6,atol=1e-8)
