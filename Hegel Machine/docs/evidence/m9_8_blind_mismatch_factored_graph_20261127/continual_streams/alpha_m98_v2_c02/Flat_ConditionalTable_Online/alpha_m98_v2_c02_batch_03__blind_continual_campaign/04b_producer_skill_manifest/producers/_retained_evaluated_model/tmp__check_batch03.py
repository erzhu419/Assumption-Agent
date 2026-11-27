import importlib.util
import pickle
import numpy as np

sp = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d, di = p['dataset'], p['campaign_diagnostics']
tr, te = d['train'], d['test']
t, dt = d['eval_times'], d['past_dt']
stress = m._stress_calibration(d['calibration_load_cells'])
xtr, _ = m._standardize(tr, te)
# The diagnostic coupons are a separately generated panel, not part of the deployment input.
_, xd = m._standardize({k: v[:0] for k, v in di.items()}, di)
train_y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
di_y = di['strain_curves_obs'] - di['strain_0'][:, None]
full = m.solve_batch(d)
print('material', full['estimated_taus'], full['hereditary_weights'])
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id'] == lab
    tr_fit = {k: v[~q] for k, v in tr.items()}
    tau, theta = m._fit(m._design(tr_fit, xtr[~q], stress, dt, t), train_y[~q], tr['instant_rate_obs'][~q])
    s, r = m._design(tr, xtr, stress, dt, t)(tau)
    e = abs((s @ theta - train_y)[q]); er = abs((r @ theta - tr['instant_rate_obs'])[q]);
    qc = di['lab_id'] == lab
    vals = []
    for chamber in (False, True):
        cs, cr = m._design(di, xd, stress, dt, t, chamber=chamber)(tau)
        ce = abs((cs @ theta - di_y)[qc]); cer = abs((cr @ theta - di['instant_rate_obs'])[qc]);
        vals.append(np.round([ce.mean(), cer.mean(), ce[:, :4].mean(), ce[:, -4:].mean()], 6).tolist())
    print('lab', lab, 'tau', tau.round(6), 'benchtop median curve/rate', np.round([np.median(e), np.median(er)], 6), 'chamber MAE curve/rate/early/late linear, saturation', vals)
cs, cr = m._design(di, xd, stress, dt, t, chamber=True)(full['estimated_taus'])
ce = abs(cs @ full['hereditary_weights'] - di_y)
cer = abs(cr @ full['hereditary_weights'] - di['instant_rate_obs'])
missing = np.isnan(di['coupon_features_raw']).any(axis=1)
for q in (~missing, missing):
    print('diagnostic feature-missing', bool(missing[q][0]), 'curve/rate MAE', ce[q].mean(), cer[q].mean())
