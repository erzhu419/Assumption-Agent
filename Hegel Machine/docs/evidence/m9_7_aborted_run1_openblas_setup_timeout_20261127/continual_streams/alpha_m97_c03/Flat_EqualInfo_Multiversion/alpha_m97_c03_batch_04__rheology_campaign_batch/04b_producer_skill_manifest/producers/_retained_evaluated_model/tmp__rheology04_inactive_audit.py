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
src = (root / 'alpha_m97_c03_batch_04_solver.py').read_text()
anchor = '    flow = np.where(act[:, None] > 0, np.log1p(times), times)'
assert src.count(anchor)==1
for name,expr in [('current',anchor),('log_base_in_low_banding','    flow = np.broadcast_to(np.log1p(times) if crystallizes else times, (len(hold), len(times)))')]:
    ns={}; exec(compile(src.replace(anchor,expr),'<audit>','exec'),ns)
    scores=[]
    for lab in np.unique(a['lab_id']):
        val=a['lab_id']==lab
        tr,te=(tuple(v[mask] for v in inputs) for mask in (~val,val))
        lt,w=ns['_fit'](tr,dt,t,crit,True,y[~val])
        err=ns['_design'](lt,te,dt,t,crit,True)@w-y[val]
        for on in (False,True):
            e=err[active[val]==on]; c=e[:,:-1]; q=e[:,-1]
            scores.append([np.sqrt(np.mean(c*c)),np.sqrt(np.mean(q*q)),np.quantile(abs(c),.9),np.quantile(abs(q),.9),np.median(c[:,-1]),np.median(q)])
    s=np.asarray(scores)
    print(name,'worst RMSE and P90 curve/rate',np.round(s[:,:4].max(axis=0),5),'late curve/rate medians by lab x gate',np.round(s[:,4:],4).tolist())
