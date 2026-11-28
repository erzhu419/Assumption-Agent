import importlib.util, pickle, numpy as np
from itertools import combinations_with_replacement
spec=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/dev_beta_m100_02_s00_solver.py'); prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(items)); item=items[0]; s=item['study']; d=item['campaign_diagnostics']; sw=s['sensor_calibration_sweeps']; a=sw['ref_rosette']; b=sw['raw_rosette']; print('calibration ranges',a.min(0),a.max(0),b.min(0),b.max(0)); print('sensors',s['sensor_ij'].tolist()); print('aux quartiles',np.quantile(s['aux_manifold_telemetry'],[0,.25,.5,.75,1],axis=0)); print('ridges, budget, cap',*[s[k] for k in ('source_ridge','nuisance_ridge','total_budget','u_box')]); print('observations rms',np.sqrt(np.mean(s['y_obs_raw']**2))); print('pulses',[(p['w_pulse'].round(4),np.quantile(p['y_pulse_raw'],[0,.5,1])) for p in d]);
for k,v in sw['single_bank_sweeps'].items(): print('single',k,np.shape(v),np.asarray(v).ravel()[:16]);
def features(x,degree):
 return np.column_stack([np.ones(len(x))]+[x[:,ix].prod(axis=1) for k in range(1,degree+1) for ix in combinations_with_replacement(range(4),k)])
S,_=prev._responses(s,prev._operator(s)); x=np.vstack([(S@p['w_pulse']).reshape(-1,4) for p in d]); y=np.vstack([p['y_pulse_raw'].reshape(-1,4) for p in d]); print('baseline ref range',x.min(0),x.max(0))
for degree in (1,2,3):
 F=features(a,degree); c=np.linalg.lstsq(F,b,rcond=None)[0]; cv=[]
 for j in range(5):
  val=np.arange(len(a))%5==j; cv.append(np.sqrt(np.mean((F[val]@np.linalg.lstsq(F[~val],b[~val],rcond=None)[0]-b[val])**2)))
 er=(features(x,degree)@c-y).reshape(len(d),-1,4); print('degree',degree,'fit rmse',np.sqrt(np.mean((F@c-b)**2)), 'max heldout fold RMSE',max(cv),'baseline pulse by pulse and bank',np.sqrt(np.mean(er**2,axis=1))); print('coef',np.round(c,6))
