import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open(root+'public_alpha_w01_batch_01.pkl','rb'))[0]
d, g = p['dataset'], p['gt']
a, b, t = d['train'], d['test'], d['eval_times']
solve = m.solve_alpha_w01_batch_01
o = solve(d)
curve = o['predicted_strain']
_, hold = m._stress(b, m._load_cells(d['calibration_load_cells']))
np.testing.assert_allclose(o['secant_compliance'], (curve[:,-1]-b['strain_0'])/np.maximum(abs(hold),.5), atol=1e-12, rtol=0)
np.testing.assert_allclose(m._crossings(g['te_clean'], b['critical_strain'], t, b['strain_0']), g['te_cross'], atol=1e-12, rtol=0)
assert np.all(np.diff(o['estimated_taus']) > 0)
print('tau max relative error', np.max(abs(o['estimated_taus']/g['taus']-1)))
for lab in range(4):
    k = b['lab_id'] == lab
    print('public lab', lab, 'strain RMSE, secant MAE, hazard MAE', np.mean((curve[k]-g['te_clean'][k])**2)**.5, np.mean(abs(o['secant_compliance'][k]-g['te_secant'][k])), np.mean(abs(o['hazard_crossing_times'][k]-g['te_cross'][k])))
# A change of lab feature units and origin must cancel through reference alignment.
moved = dict(d)
for part in ('train', 'test'):
    block = d[part]; lab = block['lab_id']; scale = np.array([1.3, .6, 2., .8])[lab,None]; shift = np.array([2., -1., .4, -2.])[lab,None]
    moved[part] = dict(block, **{k: block[k]*scale+shift for k in ('ref_coupons','coupon_features_raw')})
for k, v in solve(moved).items():
    np.testing.assert_allclose(v, o[k], atol=2e-6, rtol=0)
print('panel affine gauge passed')
# Sparse/EMI perturbation: one clean spot on lab 2, one bad reading on lab 3.
c = d['calibration_load_cells']; ids = c['lab_id']; keep = ids != 2; keep[np.flatnonzero(ids==2)[0]] = True
checks = {k: v[keep].copy() for k,v in c.items()}
checks['true_stress'][np.flatnonzero(checks['lab_id']==3)[0]] += 1.
stress = solve(dict(d, calibration_load_cells=checks))
err = np.max(abs(stress['predicted_strain'] - curve), axis=1)
assert np.max(err) < .01
print('sparse and EMI worst strain change', np.max(err))
