import importlib.util
import pickle
import numpy as np
from scipy.spatial import Delaunay

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
cal = d['calibration_load_cells']
raw, loads = cal['bridge_raw'], cal['true_channel_loads']
x = np.column_stack((np.ones(len(raw)), raw))
beta = s._bridge(raw, loads)
res = x @ beta - loads.sum(axis=1)
print('calibration noise RMS', np.sqrt(np.mean(res**2)))
sweeps = cal['single_channel_sweeps']
print('sweep structure', type(sweeps).__name__, list(sweeps) if isinstance(sweeps, dict) else np.shape(sweeps))
_, xp = np.linalg.qr(x, mode='reduced')
hull = Delaunay(raw)
coef_cov = np.linalg.inv(xp.T @ xp) * (res @ res) / (len(res) - x.shape[1])
for split in ('train', 'test'):
    for field in ('past_channel_raw', 'hold_channel_raw'):
        rows = d[split][field].reshape(-1, 4)
        design = np.column_stack((np.ones(len(rows)), rows))
        outside = hull.find_simplex(rows) < 0
        se = np.sqrt(np.einsum('ij,jk,ik->i', design, coef_cov, design))
        print(split, field, 'outside calibration hull', int(outside.sum()), '/', len(rows), 'max fitted mean SE', float(se.max()))
# Recovery with real coupon inputs and known noise-free responses tests against
# optimizer basin failures, not against unobserved deployment outcomes.
tr, te = d['train'], d['test']
xt, xe = s._features(tr, te)
p, h, f = s._inputs(tr, xt, beta)
dt, t = d['past_dt'], d['eval_times']
weights = s.solve_batch(d)['hereditary_weights']
for truth in (np.array([.065, .28]), np.array([.16, 20.]), np.array([1.1, 1.55]), np.array([4., 21.])):
    c, r = s._design(truth, p, h, f, dt, t)
    tau, theta = s._fit(p, h, f, dt, t, c @ weights, r @ weights)
    c, r = s._design(truth, *s._inputs(te, xe, beta), dt, t)
    cp, rp = s._design(tau, *s._inputs(te, xe, beta), dt, t)
    print('truth/fitted taus', truth, tau, 'max transfer curve/rate discrepancy', np.max(abs(cp @ theta - c @ weights)), np.max(abs(rp @ theta - r @ weights)))
