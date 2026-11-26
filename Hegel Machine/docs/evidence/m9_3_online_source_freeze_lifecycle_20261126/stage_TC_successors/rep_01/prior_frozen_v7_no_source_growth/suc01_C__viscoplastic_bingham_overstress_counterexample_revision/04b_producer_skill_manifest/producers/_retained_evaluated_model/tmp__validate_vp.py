import importlib.util
import pickle
import time
import numpy as np
p='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s=importlib.util.spec_from_file_location('candidate',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for i,row in enumerate(pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))):
    d,g=row['dataset'],row['gt']; start=time.monotonic(); out=m.solve_viscoplastic_hereditary_revision(d); print('runtime',time.monotonic()-start)
    print('taus',out['estimated_taus'],'rel',abs(out['estimated_taus']/g['taus']-1), 'vp',out['viscoplastic_weights'],'rel',np.linalg.norm(out['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
    q=d['test']; h=np.max(abs(q['past_stress']),axis=1)>d['yield_stress']; s=abs(q['stress_hold_0'])>d['yield_stress']; err=abs(out['predicted_strain']-g['te_clean']); missing=np.any(~np.isfinite(q['coupon_features_raw']),axis=1)
    for label,mask in [('all',h|~h),('over',h|s),('sub',~(h|s)),('over+', (h|s)&(q['stress_hold_0']>0)),('over-',(h|s)&(q['stress_hold_0']<0)),('missing coupon',missing)]+[(f'lab {j}',q['lab_id']==j) for j in np.unique(q['lab_id'])]:
        if mask.any(): print(label,mask.sum(),float(err[mask].mean()),'last',float(err[mask,-1].mean()))
