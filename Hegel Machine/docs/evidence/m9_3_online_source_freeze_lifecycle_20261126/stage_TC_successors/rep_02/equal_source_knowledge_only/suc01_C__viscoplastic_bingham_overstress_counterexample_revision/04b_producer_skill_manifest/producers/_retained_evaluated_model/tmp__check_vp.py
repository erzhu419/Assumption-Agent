import importlib.util, pickle, time
import numpy as np
p = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
for i, item in enumerate(pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))):
    d, g = item['dataset'], item['gt']
    start = time.monotonic()
    o = m.solve_viscoplastic_hereditary_revision(d)
    te = d['test']; y = d['yield_stress']; hot = (abs(te['stress_hold_0'])>y) | (abs(te['past_stress']).max(axis=1)>y)
    e = abs(o['predicted_strain'] - g['te_clean'])
    print(i, 'tau', o['estimated_taus'], 'truth',g['taus'], 'vp',o['viscoplastic_weights'],'truth',g['vp_theta'], 'mae overall/hot/cold', [e.mean(),e[hot].mean(),e[~hot].mean()], 'time',round(time.monotonic()-start,2),flush=True)
