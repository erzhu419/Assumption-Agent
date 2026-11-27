import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
t, dt = d['eval_times'], d['past_dt']
train, test = m._prepare(d)
y = np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:, None], tr['instant_rate_obs']))
cal = d['calibration_load_cells']
z, raw = cal['true_channel_loads'], cal['bridge_raw']
coef = m._fit_bridge(cal)
keep = np.linalg.norm(m._bridge_basis(z) @ coef - raw, axis=1) < .003
fold = np.empty(len(z), int)
fold[np.random.default_rng(816).permutation(len(z))] = np.arange(len(z)) % 5
err = np.full(len(z), np.nan)
for k in range(5):
    fit = fold != k
    hold = ~fit & keep
    b = m._fit_bridge(dict(true_channel_loads=z[fit], bridge_raw=raw[fit]))
    err[hold] = (m._decode_bridge(raw[hold], b) - z[hold]).sum(1)
order = np.flatnonzero(keep)[np.argsort(z[keep].sum(1))]
worst = max(np.sqrt(np.mean(err[q]**2)) for q in np.array_split(order, 10))
print('bridge CV pooled/worst decile', np.sqrt(np.nanmean(err**2)), worst)
assert worst < .001
for split in (tr, te):
    for key in ('past_channel_raw', 'hold_channel_raw'):
        r = split[key].reshape(-1, 4)
        assert np.max(abs(m._bridge_basis(m._decode_bridge(r, coef)) @ coef - r)) < 1e-8
out = m.solve_batch(d)
taus, theta = out['estimated_taus'], out['hereditary_weights']
print('full taus', taus, 'theta', theta)
assert 0.05 < taus[0] < taus[1] < 25
pred = out['predicted_strain'] - te['strain_0'][:, None]
creep0, rate = m._chamber_predict(test, np.array([0., 1e-6]), taus, theta)
assert np.allclose(creep0[:, 0], 0)
assert np.allclose(creep0[:, 1] / 1e-6, out['initial_rate_pred'], rtol=1e-5, atol=1e-9)
assert np.allclose(pred, m._chamber_predict(test, t, taus, theta)[0])
assert np.allclose(pred, m._chamber_predict((np.zeros_like(test[0]), *test[1:]), t, taus, theta)[0])
for loss in ('cauchy', 'soft_l1', 'linear'):
    resid = np.empty_like(y)
    fits = []
    for lab in np.unique(tr['lab_id']):
        held = tr['lab_id'] == lab
        a, b = m._fit(tuple(v[~held] for v in train), dt, t, y[~held], loss=loss)
        fits.append(a)
        resid[held] = m._design(tuple(v[held] for v in train), dt, t, a) @ b - y[held]
    print(loss, 'fold taus', np.round(fits, 5).tolist())
    for label, groups in [('strain', [slice(i,i+4) for i in (0,4,8)]), ('rate', [12])]:
        slices = [np.abs(resid[tr[name] == g, col]) for col in groups for name in ('lab_id', 'stratum') for g in np.unique(tr[name])]
        print(label, 'worst slice median / 90th', max(np.median(a) for a in slices), max(np.quantile(a, .9) for a in slices))
