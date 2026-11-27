import importlib.util, pickle
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
for name in ('train','chamber_verification_trials','test'):
 a=d[name]; q=a.get('aux_chamber_telemetry'); print('telemetry',name,type(q).__name__,str(q)[:950])
 print('lab counts',np.unique(a['lab_id'],return_counts=True),'missing rows',np.sum(np.any(~np.isfinite(a['coupon_features_raw']),axis=1)))
b=m._calibrate(d['calibration_load_cells']); dt=d['past_dt']; t=d['eval_times']; p=d['chamber_verification_trials']; tr=d['train']; train=m._prepare(tr,b); valid=m._prepare(p,b)
def subset(a,mask): return {k:v[mask] for k,v in a.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(mask)}
def stats(e): return np.median(abs(e[:,0])),np.median(abs(e[:,1:])),np.median(abs(e[:,-1]))
for source,target in [('pilot','chamber'),('blind','bench')]:
 g=p if source=='pilot' else tr; errors=[]
 for lab in np.unique(g['lab_id']):
  keep=g['lab_id']!=lab
  a=m._prepare(subset(g,keep),b); v=m._prepare(subset(g,~keep),b)
  tau,theta=m._estimate(train,a,dt,t) if source=='pilot' else m._estimate(a,None,dt,t)
  # The whole held-out lab forms its own reference panel. Imputation and fit do not use its response.
  e=m._design(v,tau,dt,t)@theta-v[4]; errors.append(e); print(source,'lab',lab,'median rate/curve/terminal',np.round(stats(e),6),'tau',tau.round(4))
 E=np.concatenate(errors); print(source,'worst lab',np.max([stats(e) for e in errors],axis=0),'pooled',stats(E))
for use in (True,False):
 out=m.solve_batch(d if use else {**d,'chamber_verification_trials':None}); assert all(np.isfinite(v).all() for v in out.values()); assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
 print('public pilot' if use else 'blind fallback','parameters',out['estimated_taus'].round(5),out['hereditary_weights'].round(5))
a=Path('/tmp/hegel_continual_archive'); print('archive entries',[(x.name,x.stat().st_size) for x in a.iterdir()] if a.is_dir() else [])
