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
xa, _ = m._align(a,b)
inputs = m._inputs(a,xa,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
active = (np.maximum(abs(inputs[0])-crit,0)**2 @ dt) > 0
y = np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    tr = tuple(v[~val] for v in inputs)
    te = tuple(v[val] for v in inputs)
    lt,w = m._fit(tr,dt,t,crit,True,y[~val])
    fit = m._design(lt,tr,dt,t,crit,True) @ w
    pred = m._design(lt,te,dt,t,crit,True) @ w
    print('lab',lab,'taus',np.round(np.exp(lt),4),'train rate quartiles',np.round(np.quantile(y[~val,-1]-fit[:,-1],[.25,.5,.75]),4))
    for on in (False,True):
        use = active[val] == on
        e = y[val][use] - pred[use]
        print('  active',on,'signed medians by horizon then rate',np.round(np.median(e,axis=0),4),'rate sign fraction',round(np.mean(e[:,-1]>0),3))
