import copy
import importlib.util
import pickle
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_w03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
case = pickle.loads((root / 'public_alpha_w03_batch_04.pkl').read_bytes())[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
x, h, p = m._prep(d)
n = len(tr['lab_id'])
t, dt, yld = d['eval_times'], d['past_dt'], d['yield_stress']
print('train fields:', sorted(tr), 'test fields:', sorted(te))
# Check the endpoint recurrence independently, including signs and threshold equality.
tau = np.array([.7, 3.2])
audit_h = np.array([-2*yld, -yld, 0., yld, 2*yld])
audit_p = np.vstack((p[n:n+2], np.full((1, len(dt)), 2*yld), np.full((1, len(dt)), yld), p[n+2:n+3]))
mem = np.zeros((len(audit_h), 2))
for j, dur in enumerate(dt):
    fac = np.exp(-dur/tau)
    mem = mem*fac + audit_p[:, j, None]*(1-fac)
c, r = m._design(tau, np.zeros((len(audit_h), 5)), audit_h, audit_p, dt, t, yld)
np.testing.assert_allclose(c[:, :, :2], (audit_h[:, None]-mem)[:, None, :]*(1-np.exp(-t[:, None]/tau)))
ov = np.array([-yld*yld, 0., 0., 0., yld*yld])
work = np.sum(np.maximum(np.abs(audit_p)-yld, 0.)**2*dt, axis=1)*audit_h
np.testing.assert_allclose(c[:, :, 7], ov[:, None]*t)
np.testing.assert_allclose(c[:, :, 8], work[:, None]*(1-np.exp(-t/tau[0])))
np.testing.assert_allclose(r[:, 7:], np.column_stack((ov, work/tau[0])))
assert np.all(c[3,:,7:] == 0.)
print('Independent signed Prony and plastic basis checks passed')

for variant in ('public', 'one_clean_spot_per_lab_on_wing_A_and_EMI_on_wing_B'):
    batch = copy.deepcopy(d)
    if variant != 'public':
        cc = batch['calibration_load_cells']; retain = np.ones(len(cc['lab_id']), bool)
        for lab in (0, 1):
            ids = np.flatnonzero(cc['lab_id'] == lab)
            retain[ids[1:]] = False
        for lab in (2, 3):
            ids = np.flatnonzero(cc['lab_id'] == lab)
            cc['true_stress'][ids[:3]] += [.17, -.23, .32]
        batch['calibration_load_cells'] = {k:v[retain] for k,v in cc.items()}
    _, bh, bp = m._prep(batch)
    o = m.solve_alpha_w03_batch_04(batch)
    ce, re = o['predicted_strain']-g['te_clean'], o['initial_rate_pred']-g['te_true_rate0']
    true_h, true_p = g['te_hold_true'], g['te_hist_true']
    stress_err = np.maximum(np.abs(bh[n:]-true_h), np.max(np.abs(bp[n:]-true_p), axis=1))
    due_hold = np.abs(true_h) > yld
    due_hist = np.max(np.abs(true_p), axis=1) > yld
    miss = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
    rmse = lambda a: np.sqrt(np.mean(a*a))
    print(variant)
    for lab in range(4):
        idx = te['lab_id'] == lab
        print('lab', lab, 'n', idx.sum(), 'max stress error', round(stress_err[idx].max(), 6), 'strain/onset/final RMSE', np.round([rmse(ce[idx]), rmse(re[idx]), rmse(ce[idx,-1])], 6), 'onset residual mean', round(re[idx].mean(), 6))
    for name, idx in [('hold-only', due_hold & ~due_hist), ('history-only', due_hist & ~due_hold), ('both', due_hold & due_hist), ('neither', ~due_hold & ~due_hist), ('negative hold yield', due_hold & (true_h < 0)), ('positive hold yield', due_hold & (true_h > 0)), ('missing', miss)]:
        if idx.any():
            print(name, idx.sum(), 'strain/onset/final RMSE', np.round([rmse(ce[idx]), rmse(re[idx]), rmse(ce[idx,-1])], 6))
