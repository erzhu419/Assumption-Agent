import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
ds, g = item['dataset'], item['campaign_diagnostics']
out = m.solve_batch(ds)
tau, beta, t = out['estimated_taus'], out['hereditary_weights'], ds['eval_times']
bridge = m._bridge(ds['calibration_load_cells'])
hold = m._invert_bridge(g['hold_channel_raw'], bridge).sum(axis=-1)
x = m._chamber_features(g)
y, r = m._chamber_response(tau, beta, x, hold, t)
e = y + g['strain_0'][:, None] - g['strain_curves_obs']
er = r - g['instant_rate_obs']
missing = np.isnan(g['coupon_features_raw']).any(axis=1)
for name, rows in [('all',np.ones(len(r),bool)),('complete',~missing),('missing',missing)]+[(f'lab {i}',g['lab_id']==i) for i in np.unique(g['lab_id'])]:
    print(name, 'n', rows.sum(), 'curve/rate rmse', np.sqrt(np.mean(e[rows]**2)), np.sqrt(np.mean(er[rows]**2)), 'max', np.max(abs(e[rows])), np.max(abs(er[rows])))
print('late/early max', np.max(abs(e[:,-1])), np.max(abs(e[:,0])))
print('fit', tau, beta)
s = 1e-6
small, r0 = m._chamber_response(tau, beta, x, hold, np.array([0.,s]))
assert np.all(small[:,0] == 0) and np.max(abs(small[:,1]/s-r0)) < 1e-5
assert 0.05 < tau[0] < tau[1] < 25
assert all(np.isfinite(v).all() for v in out.values())
