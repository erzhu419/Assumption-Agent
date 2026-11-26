import importlib.util, pickle, time
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for case in pickle.load(open(root+'public_suc01_C_datasets.pkl', 'rb')):
 d,g=case['dataset'],case['gt']; t0=time.monotonic(); p=m.solve_viscoplastic_hereditary_revision(d); print('seconds',time.monotonic()-t0)
 te=d['test']; h=te['stress_hold_0']; s=te['past_stress']; y=d['yield_stress']; hold=np.abs(h)>y; hist=(np.abs(s)>y).any(1); missing=~np.isfinite(te['coupon_features_raw']).all(1)
 e=np.abs(p['predicted_strain']-g['te_clean'])
 for mask,name in [(np.ones(len(h),bool),'all'), (hold|hist,'over'),(~hold&~hist,'sub'),(hold&~hist,'hold only'),(hist&~hold,'history only'),(hist&hold,'both'),(missing,'missing'),(~missing,'complete')]:
  if mask.any(): print(name,mask.sum(),'MAE',round(e[mask].mean(),6), 'late',round(e[mask,-1].mean(),6))
 print('labs',[(l,round(e[te['lab_id']==l].mean(),6)) for l in np.unique(te['lab_id'])]); print('taus',p['estimated_taus'],'rel',abs(p['estimated_taus']/g['taus']-1)); print('vp',p['viscoplastic_weights'],'rel',np.linalg.norm(p['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
 print('rate noise MAD',np.median(abs(te['instant_rate_obs']-g['te_true_rate0'])), 'max',np.max(abs(te['instant_rate_obs']-g['te_true_rate0'])))
