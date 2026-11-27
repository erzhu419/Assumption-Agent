import importlib.util, pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with (root/'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
a, b = d['train'], d['test']
out = m.solve_alpha_m97_c03_batch_04(d)
labs = b['lab_id']
complete = np.isfinite(b['coupon_features_raw']).all(axis=1)
cal = np.column_stack(m._calibrate(d['calibration_load_cells']))
g, c = cal[labs].T
hist = g[:, None]*b['past_stress_cmd'] + c[:, None]
hold = g*b['stress_hold_cmd'] + c
dt, t = d['past_dt'], d['eval_times']
damage = (np.maximum(0., np.abs(hist)-d['crit_stress'])**2) @ dt
act = np.tanh(damage)
_, x = m._align(a,b)

def direct(tau, w, rw):
    z = hist @ ((-np.expm1(-dt[:,None]/tau))*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau))
    gap = hold[:,None]-z
    coeff = w[2]+x@w[3:]+rw[1]*act*z[:,0]
    amp = gap*w[:2]
    amp[:,0] += rw[0]*act*hold
    return b['strain_0'][:,None] + amp @ (-np.expm1(-t[None,:]/tau[:,None])) + coeff[:,None]*np.log1p(t), amp @ (1/tau) + coeff

cv, rt = direct(out['estimated_taus'],out['hereditary_weights'],out['regime_mode_weights'])
np.testing.assert_allclose(cv,out['predicted_strain'],atol=1e-11,rtol=1e-11)
np.testing.assert_allclose(rt,out['initial_rate_pred'],atol=1e-11,rtol=1e-11)
print('Independent closed-form memory and curve/rate handoff agree.')
print('Stress reconstruction max abs, history/hold:',np.max(abs(hist-gt['te_hist_true'])),np.max(abs(hold-gt['te_hold_true'])))
print('Max relative tau error:', np.max(abs(out['estimated_taus']/gt['taus']-1)))
comparison = [('fitted',cv,rt),('public reference parameters with measured inputs',*direct(gt['taus'],gt['theta'],gt['vp_theta']))]
for label, yy, rr in comparison:
    e, r = yy-gt['te_clean'], rr-gt['te_true_rate0']
    print(label)
    for name, q in [('complete',complete),('missing',~complete)]+[(f'complete lab {i} damage {p}',complete&(labs==i)&((damage>0)==p)) for i in range(4) for p in (False,True)]:
        if q.any():
            print(name,int(q.sum()),'strain/rate RMSE',np.round([np.sqrt(np.mean(e[q]**2)),np.sqrt(np.mean(r[q]**2))],6),'signed terminal/rate',np.round([np.mean(e[q,-1]),np.mean(r[q])],6))
    print('missing residual unexplained by tail alone, RMS:', np.sqrt(np.mean((e[~complete]-r[~complete,None]*np.log1p(t))**2)))
