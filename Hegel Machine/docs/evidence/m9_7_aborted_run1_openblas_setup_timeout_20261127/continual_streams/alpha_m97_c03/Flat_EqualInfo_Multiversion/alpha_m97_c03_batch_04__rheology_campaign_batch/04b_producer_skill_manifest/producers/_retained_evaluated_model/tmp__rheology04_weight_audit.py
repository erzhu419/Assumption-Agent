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
anchor = '        sqrt_weight = (1 / np.hypot(1, resid / scale))[valid]'
assert src.count(anchor)==1
# Both alternatives keep the same physical response and analytic rate.
variants = {'current': '', 'common_curve_scale': '        scale[:-1] = np.median(scale[:-1])\n', 'common_curve_scale_rate_balance': '        scale[:-1] = np.median(scale[:-1])\n'}
for name, change in variants.items():
    code = src.replace(anchor, change + anchor)
    if name.endswith('rate_balance'):
        code = code.replace(anchor, anchor + '\n        sqrt_weight *= np.where(np.broadcast_to(np.arange(y.shape[1]) == y.shape[1]-1, y.shape)[valid], np.sqrt(3), 1)')
    ns = {}; exec(compile(code, str(root / 'alpha_m97_c03_batch_04_solver.py'), 'exec'),ns)
    errs=[]; taus=[]
    for lab in np.unique(a['lab_id']):
        val = a['lab_id'] == lab
        tr, te = (tuple(v[mask] for v in inputs) for mask in (~val,val))
        lt,w = ns['_fit'](tr,dt,t,crit,True,y[~val]); taus.append(np.exp(lt))
        r = ns['_design'](lt,te,dt,t,crit,True) @ w - y[val]
        for on in (False,True):
            e=r[active[val]==on]; c=e[:,:-1]; q=e[:,-1]
            errs.append([np.sqrt(np.mean(c*c)),np.sqrt(np.mean(q*q)),np.quantile(abs(c),.9),np.quantile(abs(q),.9),abs(np.median(c)),abs(np.median(q))])
    np.set_printoptions(precision=5)
    print(name,'worst curve/rate pairs: RMSE, P90AE, abs median bias',np.max(errs,axis=0),'tau min/max',np.min(taus,axis=0),np.max(taus,axis=0))
    print('per lab x regime',np.asarray(errs)[:,2:])
