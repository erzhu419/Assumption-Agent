import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, _ = m._standardize(tr, te)
stress = m._stress_calibration(d['calibration_load_cells'])
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
full = m.solve_batch(d)
ref_tau = full['estimated_taus']

def slice_site(mask):
    return {k: tr[k][mask] for k in ('past_channel_raw', 'hold_channel_raw')}

scores = []
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id'] == lab
    tau, th = m._fit(m._design(slice_site(~q), x[~q], stress, dt, t), y[~q], r[~q])
    design, rate = m._design(slice_site(q), x[q], stress, dt, t)(tau)
    v = [np.max(abs(tau/ref_tau - 1)), np.median(abs(y[q] - design @ th)), np.median(abs(r[q] - rate @ th))]
    scores.append(v)
    print('held-out train lab', lab, 'n', q.sum(), '[max relative tau shift, median strain error, median onset error]', np.round(v, 7))
print('worst held-out train lab:', np.round(np.max(scores, axis=0), 7))

# Independently evaluate the chronological recurrence on every measured history.
for site in (tr, te):
    past = stress(site['past_channel_raw'])
    z = np.zeros((len(past), 2))
    for j, step in enumerate(dt):
        a = np.exp(-step/ref_tau)
        z = a*z + (1-a)*past[:, j, None]
    gap = stress(site['hold_channel_raw'])[:, None] - z
    _, onset = m._design(site, np.zeros((len(past), 4)), stress, dt, t)(ref_tau)
    np.testing.assert_allclose(onset[:, :2], gap/ref_tau, rtol=1e-12, atol=1e-12)
print('chronological recurrence matches deployed design on all train/test histories')
