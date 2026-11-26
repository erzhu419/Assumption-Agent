import importlib.util, pickle, time
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('cand', root+'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p, = pickle.load(open(root+'public_suc01_C_datasets.pkl', 'rb'))
d, g = p['dataset'], p['gt']
t = time.perf_counter()
o = m.solve_viscoplastic_hereditary_revision(d)
print('seconds', time.perf_counter()-t)
print('tau', o['estimated_taus'], 'rel', abs(o['estimated_taus']/g['taus']-1))
print('vp', o['viscoplastic_weights'], 'rel L2', np.linalg.norm(o['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
s = d['test']; y = d['yield_stress']; over = (abs(s['stress_hold_0']) > y) | (abs(s['past_stress']).max(axis=1)>y)
e = abs(o['predicted_strain']-g['te_clean'])
print('MAE all / over / sub', e.mean(), e[over].mean(), e[~over].mean())
for name in ('lab_id', 'stratum'):
 print(name, [(int(i),float(e[s[name] == i].mean())) for i in np.unique(s[name])])
print('MAE by time', e.mean(axis=0))
