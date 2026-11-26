import runpy
import numpy as np

v = runpy.run_path('/tmp/score_viscoplastic.py')
m, d, g, out = (v[k] for k in ('m', 'd', 'g', 'o'))
s, threshold = d['test'], d['yield_stress']
dt, time = d['past_dt'], d['eval_times']
held = abs(s['stress_hold_0']) > threshold
past = (abs(s['past_stress']) > threshold).any(axis=1)
history = m._history(s, dt, threshold)
np.testing.assert_allclose(history[2], np.sign(s['stress_hold_0']) * np.maximum(abs(s['stress_hold_0']) - threshold, 0)**2)
np.testing.assert_allclose(history[3], s['stress_hold_0'] * np.sum(dt * np.maximum(abs(s['past_stress']) - threshold, 0)**2, axis=1))
x = np.zeros((len(held), 4))
tau = out['estimated_taus']
eps = 1e-6
c, r = m._design(history, dt, np.array([0., eps]), x, tau)
np.testing.assert_allclose(c[:, 0], 0., atol=1e-12)
np.testing.assert_allclose(c[:, 1] / eps, r, rtol=5e-6, atol=1e-9)
assert not np.any(c[~held, :, -2]) and not np.any(c[~past, :, -1])

design = m._design

def ablated(history, dt, time, x, taus):
    curve, rate = design(history, dt, time, x, taus)
    return curve[..., :-2], rate[..., :-2]

m._design = ablated
try:
    baseline = m.solve_viscoplastic_hereditary_revision(d)['predicted_strain']
finally:
    m._design = design

a, b = (abs(pred - g['te_clean']) for pred in (baseline, out['predicted_strain']))
for name, mask in [('sub-yield', ~(held | past)), ('hold only', held & ~past), ('history only', past & ~held), ('both', held & past), ('positive hold', held & (s['stress_hold_0'] > 0)), ('negative hold', held & (s['stress_hold_0'] < 0))]:
    if mask.any():
        print(name, int(mask.sum()), 'MAE Prony-only / full', np.round([a[mask].mean(), b[mask].mean()], 5), 'full last', round(b[mask, -1].mean(), 5))
for lab in np.unique(s['lab_id']):
    for label, cohort in [('over', held | past), ('sub', ~(held | past))]:
        mask = (s['lab_id'] == lab) & cohort
        if mask.any():
            print('lab', int(lab), label, int(mask.sum()), 'MAE Prony-only / full', np.round([a[mask].mean(), b[mask].mean()], 5))
assert b.mean() <= .070 and b[held | past].mean() <= .080 and b[~(held | past)].mean() <= .065
assert np.max(abs(tau / g['taus'] - 1)) <= .10
assert np.linalg.norm(out['viscoplastic_weights'] - g['vp_theta']) <= .14 * np.linalg.norm(g['vp_theta'])
