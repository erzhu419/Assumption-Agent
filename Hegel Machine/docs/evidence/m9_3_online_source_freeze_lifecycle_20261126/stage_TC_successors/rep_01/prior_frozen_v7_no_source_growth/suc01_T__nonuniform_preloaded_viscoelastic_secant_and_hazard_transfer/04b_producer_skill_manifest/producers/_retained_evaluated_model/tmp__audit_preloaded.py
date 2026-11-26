import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = pickle.load(open(root + 'public_suc01_T_datasets.pkl', 'rb'))[0]
d, gt = b['dataset'], b['gt']
a, test, dt, t = d['train'], d['test'], d['past_dt'], d['eval_times']
tau, amp, drift = m._fit_physics(a, dt, t)
curves, rate = m._branch(tau, a, dt, t)
tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
z = a['initial_memory_states'] * np.exp(-np.sum(dt) / tau) + a['past_stress'] @ (-np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau))
gap = a['stress_hold_0'][:, None] - z
np.testing.assert_allclose(rate, gap / tau, atol=1e-12)
np.testing.assert_allclose(curves, gap[:, None, :] * (1 - np.exp(-t[:, None] / tau)), atol=1e-12)
eps = 1e-6
short, _ = m._branch(tau, a, dt, np.array([eps]))
np.testing.assert_allclose(short[:, 0] @ amp / eps + drift, rate @ amp + drift, rtol=1e-5, atol=1e-5)
print('Train worst-horizon median absolute curve residual, measured versus zero preload (same fitted parameters):')
obs = a['strain_curves_obs'] - a['strain_0'][:, None]
for c in (curves, m._branch(tau, {**a, 'initial_memory_states': np.zeros_like(a['initial_memory_states'])}, dt, t)[0]):
    print(np.median(np.abs(obs - c @ amp - drift[:, None]*t), axis=0).max())
x, known, weights = m._coupons(a, test)
n = len(drift)
x, known = x[:n], known[:n].all(axis=1)
lab, stratum = a['lab_id'], a['stratum']
base = np.column_stack([np.ones(n), x])
quadratic = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)])
rat = np.column_stack([x[:, i]/(1 + x[:, j]**2) for i in range(4) for j in range(4)])
# Holding out one cell per lab permits calibration on other strata in that lab.
lab_design = (base[:, :, None] * np.eye(4)[lab, None, :]).reshape(n, -1)
for label, F in [('shared affine',base), ('lab calibrated affine',lab_design), ('lab calibrated + quadratic/rational', np.column_stack([lab_design,quadratic,rat]))]:
    scores = []
    for l in np.unique(lab):
        for s in np.unique(stratum):
            held = known & (lab==l) & (stratum==s)
            keep = known & ~held
            sw = np.sqrt(weights[keep]); coef = np.linalg.lstsq(F[keep]*sw[:, None], drift[keep]*sw, rcond=None)[0]
            scores.append(np.mean(np.abs(drift[held] - F[held]@coef)))
    print(label, 'mean/worst held-cell drift MAE', np.mean(scores), np.max(scores))
o = m.solve_preloaded_hereditary_transfer(d)
mae = np.column_stack([np.abs(o['predicted_strain'] - gt['te_clean']).mean(1), np.abs(o['secant_compliance'] - gt['te_secant']), np.abs(o['hazard_crossing_times'] - gt['te_cross'])])
preload = np.linalg.norm(test['initial_memory_states'] * np.exp(-sum(dt)/o['estimated_taus']), axis=1)
for q in np.array_split(np.argsort(preload), 4):
    print('preload quartile n/max memory / strain secant crossing MAE', len(q), preload[q].max(), mae[q].mean(0))
