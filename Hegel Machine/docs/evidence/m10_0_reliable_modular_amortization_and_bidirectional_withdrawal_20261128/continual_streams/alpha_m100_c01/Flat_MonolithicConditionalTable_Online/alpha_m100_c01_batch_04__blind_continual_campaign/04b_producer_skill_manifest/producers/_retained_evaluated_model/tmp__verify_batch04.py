import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = p['dataset']; tr, te = d['train'], d['test']; dt, times = d['past_dt'], d['eval_times']
x, xt = m._features(tr, te)
cal = d['calibration_load_cells']; b, keep = m._fit_bridge(cal['bridge_raw'], cal['true_channel_loads'])
print('bridge rejected rows:', np.flatnonzero(~keep))
h, s, f = m._calibrated_inputs(tr, x, b)
y, r = tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']
full = m.solve_batch(d); tau, theta = full['estimated_taus'], full['hereditary_weights']
print('full taus, theta:', tau, theta)
assert set(full) == {'estimated_taus', 'hereditary_weights', 'predicted_strain', 'initial_rate_pred'}
assert full['predicted_strain'].shape == (len(xt), len(times)) and full['initial_rate_pred'].shape == (len(xt),)
assert all(np.isfinite(a).all() for a in full.values()) and .05 < tau[0] < tau[1] < 25
for lab in np.unique(tr['lab_id']):
    v = tr['lab_id'] == lab; j = ~v
    t, w = m._fit(h[j], s[j], f[j], dt, times, y[j], r[j])
    c, a = m._design(t, h[v], s[v], f[v], dt, times)
    e, er = np.abs(c@w-y[v]), np.abs(a@w-r[v])
    print('held-out lab', lab, 'taus', t, 'strain abs p50/p90', np.quantile(e, [.5,.9]), 'rate abs p50/p90', np.quantile(er,[.5,.9]))
_, sh, ff = m._calibrated_inputs(te, xt, b)
eps = 1e-6
zero, _ = m._predict_chamber(tau, theta, sh, ff, np.array([0.0, eps]), m._CHAMBER_PROJECTION)
assert np.max(np.abs(zero[:, 0])) < 1e-12
err = np.max(np.abs(zero[:, 1] / eps - full['initial_rate_pred']))
print('chamber initial-rate derivative max error:', err)
assert err < 1e-5

def walk(item):
    if isinstance(item, dict):
        if 'facility_telemetry_trace' in item:
            a = np.asarray(item['facility_telemetry_trace']); print('trace', {k:v for k,v in item.items() if not isinstance(v,(dict,list))}, 'medians', np.median(a, axis=0).round(4))
        for k,v in item.items():
            if k != 'facility_telemetry_trace': walk(v)
    elif isinstance(item, list):
        for v in item: walk(v)
walk(json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')))
