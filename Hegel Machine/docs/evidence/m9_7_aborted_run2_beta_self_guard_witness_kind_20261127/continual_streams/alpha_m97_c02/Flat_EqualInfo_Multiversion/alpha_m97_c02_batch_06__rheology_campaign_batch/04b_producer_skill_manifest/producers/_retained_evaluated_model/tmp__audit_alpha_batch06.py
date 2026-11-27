import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c02_batch_06.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
tr, te = d['train'], d['test']
cal = m._calibration(d)
c = d['calibration_load_cells']
for lab in range(4):
    rows = c['lab_id'] == lab
    x, y = c['cmd_stress'][rows], c['true_stress'][rows]
    residual = y - cal[lab, 0]*x - cal[lab, 1]
    clean = abs(residual) <= 3
    print('channel', lab, 'clean/spikes', clean.sum(), (~clean).sum(), 'max clean residual', max(abs(residual[clean])))
    if clean.sum() >= 4:
        np.testing.assert_allclose(cal[lab], np.linalg.lstsq(np.c_[x[clean], np.ones(clean.sum())], y[clean], rcond=None)[0], atol=1e-12, rtol=0)
np.testing.assert_allclose(cal, np.c_[g['stress_gains'],g['stress_biases']], atol=1e-4,rtol=0)
_, aligned = m._aligned(tr, te)
out = m.solve_alpha_m97_c02_batch_06(d)
_, hold, _ = m._inputs(te, aligned, cal)
complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
t = d['eval_times']
phi = t/(1+0.85*t)

def rmse(e):
    return float(np.sqrt(np.mean(e*e)))

# Reference parameters are used only to separate alignment error from fit error.
past, _, _ = m._inputs(te, aligned, cal)
tau = g['taus']
z = np.zeros((len(hold), 2))
for stress, dt in zip(past.T, d['past_dt']):
    decay = np.exp(-dt/tau)
    z = z*decay + stress[:, None]*(1-decay)
gap = hold[:, None]-z
hw, rw = g['theta'], g['vp_theta']
exp_terms = -np.expm1(-t[:, None]/tau)
flow = hw[2]+aligned @ hw[3:7]+rw[0]*np.sign(hold)*np.sqrt(abs(hold))
contrast = rw[1]*(gap[:,0]-gap[:,1])
ref_curve = te['strain_0'][:, None]+(gap*hw[:2]) @ exp_terms.T+flow[:, None]*phi+contrast[:, None]*exp_terms[:,0]
ref_rate = (gap/tau) @ hw[:2]+flow+contrast/tau[0]
err = out['predicted_strain']-g['te_clean']
rate_err = out['initial_rate_pred']-g['te_true_rate0']
for name, mask in [('negative hold, complete', complete & (hold < 0)), ('nonnegative hold, complete', complete & (hold >= 0)), ('imputed', ~complete)]:
    assert mask.any(), name
    print(name, 'n', mask.sum(), 'fit early/late/rate RMSE', rmse(err[mask,:3]), rmse(err[mask,-3:]), rmse(rate_err[mask]), 'reference-model early/late/rate RMSE', rmse((ref_curve-g['te_clean'])[mask,:3]), rmse((ref_curve-g['te_clean'])[mask,-3:]), rmse((ref_rate-g['te_true_rate0'])[mask]))
    if name != 'imputed':
        assert max(rmse(err[mask,:3]), rmse(err[mask,-3:]), rmse(rate_err[mask])) < 0.02
r = ref_curve-g['te_clean']
ref_rate_error = ref_rate-g['te_true_rate0']
print('reference residual left after removing shared drift on imputed rows', rmse((r-ref_rate_error[:, None]*phi)[~complete]))
assert rmse((r-ref_rate_error[:, None]*phi)[~complete]) < 0.002
