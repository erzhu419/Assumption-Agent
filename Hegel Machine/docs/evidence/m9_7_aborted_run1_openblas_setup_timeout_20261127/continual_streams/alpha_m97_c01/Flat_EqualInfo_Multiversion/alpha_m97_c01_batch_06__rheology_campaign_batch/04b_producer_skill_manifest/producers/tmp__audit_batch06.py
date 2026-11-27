import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c01_batch_06.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
out = m.solve_alpha_m97_c01_batch_06(d)
te = d['test']; t = np.asarray(d['eval_times']); dt = np.asarray(d['past_dt'])
past, hold, x = (a[-len(te['lab_id']):] for a in m._prepare(d))
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
lag = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
z = past @ (np.exp(-lag[:,None]/tau) * -np.expm1(-dt[:,None]/tau))
g = hold[:,None] - z
flow = hw[2] + x @ hw[3:] + rw[0]*np.sign(hold)*np.sqrt(abs(hold))
fast = hw[0]*g[:,0] + rw[1]*(g[:,0]-g[:,1]); slow = hw[1]*g[:,1]
def creep(t):
    return -fast[:,None]*np.expm1(-t/tau[0])-slow[:,None]*np.expm1(-t/tau[1])+flow[:,None]*t/(1+.85*t)
np.testing.assert_allclose(te['strain_0'][:,None]+creep(t), out['predicted_strain'], atol=1e-12)
h = np.array([1e-5]); numerical_rate = ((creep(h)-creep(-h))/(2*h))[:,0]
np.testing.assert_allclose(numerical_rate, out['initial_rate_pred'], atol=1e-8)
print('Independent closed-form curve and centered initial derivative agree')
for key in ('taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases'):
    print('public reference', key, np.asarray(gt[key]).shape, gt[key])
print('estimated taus and weights', tau, hw, rw)
cal = m._calibrate(d['calibration_load_cells'])
print('calibration max abs coefficient errors', abs(cal-np.column_stack((gt['stress_gains'],gt['stress_biases']))).max(axis=0))
e = out['predicted_strain'] - gt['te_clean']; er = out['initial_rate_pred'] - gt['te_true_rate0']
print('clean pooled MAE curve/rate', abs(e).mean(), abs(er).mean())
for field, levels in [('lab_id', te['lab_id']), ('stratum', te['stratum']), ('hold_sign', np.sign(hold))]:
    for level in np.unique(levels):
        use = levels == level
        hmae = abs(e[use]).mean(axis=0)
        print(field, level, 'n', use.sum(), 'curve worst-horizon MAE', round(hmae.max(),6), 'horizon', t[hmae.argmax()], 'rate MAE', round(abs(er[use]).mean(),6), 'signed terminal/rate bias', np.round([e[use,-1].mean(), er[use].mean()],6))
