import copy
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c02_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
solve = m.solve_alpha_m97_c02_batch_04
out = solve(d)
for v in out.values():
    assert np.isfinite(v).all()
assert np.all(np.diff(out['estimated_taus']) > 0)
print('taus, hereditary and regime weights:', *(out[k].round(6).tolist() for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights')))
training, test = d['train'], d['test']
n = len(training['lab_id'])
dt, ts = d['past_dt'], d['eval_times']
prepared = m._prepare(d)
aligned, past, hold, act = (v[n:] for v in prepared)
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
z = np.zeros((len(hold),2))
for step, duration in enumerate(dt):
    decay = np.exp(-duration/tau)
    z = decay*z + (1-decay)*past[:, step, None]
act_independent = np.tanh(np.sum(dt*np.maximum(np.abs(past)-d['crit_stress'], 0)**2, axis=1))
np.testing.assert_allclose(act, act_independent, rtol=1e-13, atol=1e-13)
assert (act == 0).any() and (act > 0).any()
gap = hold[:, None]-z
drift = hw[2]+aligned@hw[3:]+rw[1]*act*z[:, 0]
def strain(t):
    e = -np.expm1(-t[:, None]/tau)
    return test['strain_0'][:,None] + (gap*hw[:2])@e.T + (rw[0]*act*hold)[:, None]*e[:, 0] + drift[:, None]*np.log1p(t)
np.testing.assert_allclose(out['predicted_strain'], strain(ts), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], (strain(np.array([1e-6]))[:,0]-test['strain_0'])/1e-6, rtol=1e-5, atol=1e-5)
changed = copy.deepcopy(d)
for split in ('train','test'):
    changed[split]['aux_chamber_telemetry'] = np.full_like(changed[split]['aux_chamber_telemetry'], 1000.)
for k,v in solve(changed).items():
    np.testing.assert_array_equal(out[k],v)
base_cal = m._calibrate(d['calibration_load_cells'])
x = np.tile([-2., -1., 0., 1., 2.], 4); lab = np.repeat(np.arange(4), 5)
g = np.array([.9, 1.1, .8, 1.3]); b = np.array([.2, -.1, .1, -.2])
ycal = g[lab]*x+b[lab]
def calibrate(y):
    return m._calibrate(dict(lab_id=lab, cmd_stress=x, true_stress=y))
np.testing.assert_allclose(calibrate(ycal), np.column_stack((g,b)), atol=1e-12)
ycal[[0,5,6]] += [7,8,-9]
np.testing.assert_allclose(calibrate(ycal)[0], [g[0],b[0]], atol=1e-12)
np.testing.assert_allclose(calibrate(ycal)[1], m._calibrate(dict(lab_id=np.array([],dtype=int),cmd_stress=[],true_stress=[]))[1], atol=1e-12)
print('Passed recurrence, initial derivative, telemetry independence, and clean/EMI calibration checks.')
lab = training['lab_id']; complete = np.isfinite(training['coupon_features_raw']).all(1)
a = tuple(v[:n] for v in prepared)
y = training['strain_curves_obs']-training['strain_0'][:,None]; r = training['instant_rate_obs']
e, er = np.empty_like(y), np.empty_like(r)
for c in range(4):
    tr, va = complete & (lab != c), lab == c
    t, w = m._fit(*(v[tr] for v in a), dt, ts, y[tr], r[tr], True)
    curve, rate = m._design(*(v[va] for v in a), dt, ts, t, True)
    e[va], er[va] = y[va]-curve@w, r[va]-rate@w
for flag in (True,False):
    groups = [(lab==c)&(complete==flag)&((a[-1]>0)==active) for c in range(4) for active in (True,False)]
    scores = [(np.quantile(abs(e[q]),.9), np.sqrt(np.mean(er[q]**2))) for q in groups if q.any()]
    print('complete',flag,'worst lab x damage: strain absolute error q90 and rate RMSE',np.max(scores,axis=0).round(5).tolist())
print('Pooled median absolute strain and rate errors:', np.median(abs(e)), np.median(abs(er)))
print('Complete zero-damage: median final-time signed residual', np.median(e[complete & (a[-1]==0),-1]))
