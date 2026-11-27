import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
tr, te = ds['train'], ds['test']
lab = tr['lab_id']
x = m._features(tr, te)[:len(lab)]
a = m._coupon_inputs(tr, x, m._calibration(ds['calibration_load_cells']))
dt, t = ds['past_dt'], ds['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
base_tau, base_th = m._fit(a, dt, t, y, r)
print('full fit tau:', base_tau)
for g in np.unique(lab):
    hold = lab == g
    tau, th = m._fit(tuple(v[~hold] for v in a), dt, t, y[~hold], r[~hold])
    C, R = m._design(tau, tuple(v[hold] for v in a), dt, t)
    e, er = C @ th - y[hold], R @ th - r[hold]
    print('held lab', g, 'n', hold.sum(), 'tau', tau, 'fractional shift', tau/base_tau-1,
          'median absolute curve/rate error', np.nanmedian(abs(e)), np.nanmedian(abs(er)),
          'signed median curve/rate error', np.nanmedian(e), np.nanmedian(er))
