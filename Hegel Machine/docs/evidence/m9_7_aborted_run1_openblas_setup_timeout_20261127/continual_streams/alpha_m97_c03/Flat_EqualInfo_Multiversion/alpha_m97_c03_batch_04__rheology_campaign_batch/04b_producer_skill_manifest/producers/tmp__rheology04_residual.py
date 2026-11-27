import importlib.util
import pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b = d['train'], d['test']
solve = m.solve_alpha_m97_c03_batch_04
xc, _ = m._align(a,b)
z = m._inputs(a,xc,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
y=np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
lt,w=m._fit(z,dt,t,crit,True,y)
r=y-m._design(lt,z,dt,t,crit,True)@w
print('absolute residual quantiles by horizon and rate (last):')
print(np.round(np.quantile(abs(r),[.25,.5,.75,.9,.95,.98,.995,1],axis=0),5))
print('largest residuals',[(int(i),int(j),round(r[i,j],3)) for i,j in zip(*np.where(abs(r)>2))]); print('per-row counts',np.unique((abs(r)>.5).sum(axis=1),return_counts=True))
