import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c02_batch_03.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
dt, times = d['past_dt'], d['eval_times']
past, hold, x = m._prepare(d)
out = m.solve_alpha_m97_c02_batch_03(d)
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
z = np.zeros((len(hold), 2))
for step, duration in enumerate(dt):
    decay = np.exp(-duration/tau)
    z = z*decay + (1-decay)*past[:,step,None]
g0, g1 = (hold[:,None]-z).T
terminal = hw[2] + x@hw[3:] + rw[0]*np.sign(hold)*np.sqrt(abs(hold))
amp0 = hw[0]*g0 + rw[1]*(g0-g1)
amp1 = hw[1]*g1
def response(t):
    return (amp0[:,None]*(-np.expm1(-t/tau[0])) + amp1[:,None]*(-np.expm1(-t/tau[1])) + terminal[:,None]*t/(1+.85*t))
np.testing.assert_allclose(out['predicted_strain'], te['strain_0'][:,None]+response(times)[n:], rtol=0, atol=2e-12)
np.testing.assert_allclose(out['initial_rate_pred'], (amp0/tau[0]+amp1/tau[1]+terminal)[n:], rtol=0, atol=2e-12)
fd = ((4*response(np.array([5e-6]))-response(np.array([1e-5])))/1e-5)[n:,0]
print('scalar recurrence and closed-form derivative agree; max finite-difference rate error', np.max(abs(fd-out['initial_rate_pred'])))
assert np.max(abs(fd-out['initial_rate_pred'])) < 1e-8

complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    fit = (tr['lab_id']!=lab)&complete
    val = (tr['lab_id']==lab)&complete
    t, w = m._fit(past[:n][fit], hold[:n][fit], x[:n][fit], dt, times, y[fit], r[fit])
    c, a = m._design(past[:n][val], hold[:n][val], x[:n][val], dt, times, t)
    e, er = y[val]-c@w, r[val]-a@w
    for stratum in np.unique(tr['stratum'][val]):
        q = tr['stratum'][val]==stratum
        print('held lab/stratum',lab,stratum,'median signed curve/rate residual',round(np.median(e[q]),6),round(np.median(er[q]),6),'median abs',round(np.median(abs(e[q])),6),round(np.median(abs(er[q])),6))

sweep = d['calibration_load_cells']; base = m._calibrate(sweep)
issue = []
for lab in range(4):
    ii = np.flatnonzero(sweep['lab_id']==lab)
    for count in (4, 5, len(ii)):
        keep = np.ones(len(sweep['lab_id']),dtype=bool); keep[ii[count:]]=False
        for point in range(count):
            for spike in (-8.,8.):
                s = {k:v[keep].copy() for k,v in sweep.items()}
                j = np.flatnonzero(s['lab_id']==lab)[point]
                s['true_stress'][j] += spike
                error = np.max(abs(m._calibrate(s)[lab]-base[lab]))
                if error > 0.002: issue.append((lab,count,point,spike,round(error,6)))
print('single-spike placement deviations > 0.002:', len(issue), 'worst:', sorted(issue, key=lambda v:v[-1], reverse=True)[:6])
