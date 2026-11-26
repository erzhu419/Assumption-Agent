import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a, _, y, complete = s._prepare(d)
complete = complete[:len(y)]
labs = d['train']['lab_id']
unique = np.unique(labs)
t = a.times
results = {k: np.empty_like(y[:, :-1]) for k in ['archived', 'missing_only', 'full_rate', 'nested_robust']}

def residual(held_labs, predict_on):
    tau, coef = s._fit(a, y, complete & ~np.isin(labs, held_labs))
    return y[predict_on] - a.matrix(tau)[predict_on] @ coef

for lab in unique:
    held = labs == lab
    e = residual([lab], held)
    inner = np.zeros_like(y)
    for other in unique[unique != lab]:
        rows = labs == other
        inner[rows] = residual([lab, other], rows)
    g = []
    for c in [False, True]:
        r = inner[(~held) & (complete == c)]
        rr = r[:, :-1]
        scale = max(1e-6, 1.4826 * np.median(np.abs(rr - np.median(rr))))
        def loss(gain):
            u = (rr - gain*r[:, -1, None]*t)/(2*scale)
            return np.log1p(u*u).sum()
        g.append(minimize_scalar(loss, bounds=(0, 1), method='bounded').x)
    print('held lab', lab, 'nested gains, missing/complete', np.round(g, 4))
    for name, gain in [('archived', np.where(complete[held], .5, 1.)), ('missing_only', (~complete[held]).astype(float)), ('full_rate', np.ones(held.sum())), ('nested_robust', np.where(complete[held], g[1], g[0]))]:
        results[name][held] = e[:, :-1] - gain[:, None]*e[:, -1, None]*t

high = a.ov != 0
masks = {'pooled': np.ones(len(y), dtype=bool), 'complete': complete, 'imputed': ~complete, 'high+': a.ov>0, 'high-': a.ov<0, 'extreme': np.abs(a.work) >= np.quantile(np.abs(a.work), .90)}
masks.update({f'lab{lab}-{reg}': (labs==lab)&rows for lab in unique for reg,rows in [('low',~high),('high',high)]})
print('columns:', *results)
for name, rows in masks.items():
    rmse = [np.sqrt(np.mean(r[rows]**2)) for r in results.values()]
    med = [np.median(np.abs(r[rows])) for r in results.values()]
    tail = [np.sqrt(np.mean(r[rows,-1]**2)) for r in results.values()]
    print(name, 'n', rows.sum(), 'RMSE', np.round(rmse,5), 'MedAE', np.round(med,5), 'tailRMSE', np.round(tail,5))
