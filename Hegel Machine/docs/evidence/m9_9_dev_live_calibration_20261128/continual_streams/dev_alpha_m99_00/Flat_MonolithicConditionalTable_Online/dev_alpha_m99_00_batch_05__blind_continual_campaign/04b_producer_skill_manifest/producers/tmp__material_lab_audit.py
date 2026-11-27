import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
lab = tr['lab_id']
x = m._features(tr, te)[:len(lab)]
bridge = m._bridge(d['calibration_load_cells'])
coupon = m._coupon(tr, x, bridge)
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
out = m.solve_batch(d)
tau0 = out['estimated_taus']
rows = []
for l in np.unique(lab):
    keep = lab != l
    tau, theta = m._fit(tuple(v[keep] for v in coupon), dt, t, y[keep], r[keep])[:2]
    A, B = m._design(tau, tuple(v[~keep] for v in coupon), dt, t)
    e, q = abs(A @ theta - y[~keep]), abs(B @ theta - r[~keep])
    row = [np.median(e[:, :4]), np.median(e[:, -4:]), np.median(q), max(abs(tau/tau0 - 1))]
    rows.append(row)
    print('held lab', l, 'early/late/rate MedAE, max relative tau shift', np.round(row, 6))
print('worst benchtop slice', np.max(rows, axis=0))
