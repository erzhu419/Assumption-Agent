import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b, y, complete = s._prepare(d)
lab = d['train']['lab_id']
complete = complete[:len(lab)]
e0 = np.empty_like(y[:, :-1]); e1 = np.empty_like(e0)
for c in np.unique(lab):
    h = lab == c
    lt, theta = s._fit(a, y, ~h & complete)
    p = a.matrix(lt)[h] @ theta
    base = y[h, :-1] - p[:, :-1]
    rate_error = y[h, -1] - p[:, -1]
    ok = complete[h]
    e0[h] = base - np.where(ok, 0., rate_error)[:, None] * a.times
    # Match deployment: infer the shared correction from rates, never held-out curves.
    drift = np.median(rate_error[ok])
    e1[h] = base - np.where(ok, drift, rate_error)[:, None] * a.times
    med = lambda e: np.median(np.abs(e))
    print('lab', c, 'median rate innovation', round(drift, 5), 'complete curve MedAE', round(med(e0[h][ok]), 5), '->', round(med(e1[h][ok]), 5))
rmse = lambda x: np.sqrt(np.mean(x*x))
high = a.ov != 0
masks = [(lab == c) & m for c in np.unique(lab) for m in (~high, high)]
for name, e in [('current', e0), ('lab_rate_centered', e1)]:
    print(name, 'worst lab/load curve RMSE', max(rmse(e[m]) for m in masks), 'worst tail RMSE', max(rmse(e[m, -1]) for m in masks), 'worst lab/load MedAE', max(np.median(np.abs(e[m])) for m in masks), 'pooled MedAE', np.median(np.abs(e)))
