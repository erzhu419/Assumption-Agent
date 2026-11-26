import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
train = d['train']
a, _, y, complete = s._prepare(d)
complete = complete[:len(y)]
labs = train['lab_id']
base = np.empty_like(y)
for lab in np.unique(labs):
    held = labs == lab
    tau, coef = s._fit(a, y, (~held) & complete)
    base[held] = a.matrix(tau)[held] @ coef
    print('held lab', lab, 'tau', np.exp(tau), 'plastic', coef[7:9], flush=True)
tau, coef = s._fit(a, y, complete)
print('full tau', np.exp(tau), 'coefficients', coef)
delta = y[:, -1] - base[:, -1]
high = a.ov != 0
extreme = np.abs(a.work) >= np.quantile(np.abs(a.work), .90)
masks = {'low': ~high, 'high+': a.ov>0, 'high-': a.ov<0, 'extreme': extreme, 'complete': complete, 'imputed': ~complete}
masks.update({f'lab{lab}-{reg}': (labs==lab)&m for lab in np.unique(labs) for reg,m in [('low',~high),('high',high)]})
raw = y[:, :-1] - base[:, :-1]
t = a.times
for name,m in masks.items():
    gains = np.stack([np.zeros(len(y)), np.where(complete, .5, 1.), np.ones(len(y))])
    err = raw[None,m,:] - gains[:,m,None]*delta[None,m,None]*t
    rmse = np.sqrt(np.mean(err**2, axis=(1,2)))
    tail = np.sqrt(np.mean(err[:,:,-1]**2, axis=1))
    dr = delta[m,None]*np.ones_like(t)
    print(f'{name:12s} n={m.sum():3d}  curve rmse {rmse.round(5)}  tail {tail.round(5)}  projected gains {((raw[m]*dr).sum(axis=0) / (dr**2).sum(axis=0) / t).round(3)}')
