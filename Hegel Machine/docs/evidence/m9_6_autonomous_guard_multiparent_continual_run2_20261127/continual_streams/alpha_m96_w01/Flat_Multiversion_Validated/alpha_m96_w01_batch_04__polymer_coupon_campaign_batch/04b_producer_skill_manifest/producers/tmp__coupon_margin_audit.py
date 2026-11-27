import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
s = importlib.util.spec_from_file_location('candidate', root + 'alpha_m96_w01_batch_04_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open(root + 'public_alpha_m96_w01_batch_04.pkl', 'rb') as f:
    p = pickle.load(f)[0]
d, g = p['dataset'], p['gt']
tr, te = d['train'], d['test']
lab = np.asarray(tr['lab_id'])
n = len(lab)
dt, t, sy = d['past_dt'], d['eval_times'], d['yield_stress']
c = m._calibration(d['calibration_load_cells'])
x, miss = m._features(tr, te)
a, b = m._stresses(tr, c)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
rm = lambda z: np.sqrt(np.mean(z**2))
# Estimate onset noise and missing-feature slope uncertainty in held-out labs.
# Project each held-out curve onto time after removing fitted Prony and plasticity.
# Temporal median is insensitive to isolated bad curve readings.
clean = ~miss[:n].any(axis=1)
for j in range(4):
    fit = clean & (lab != j)
    k = clean & (lab == j)
    tau, th = m._fit(a[fit], b[fit], dt, t, x[:n][fit], sy, y[fit], r[fit])
    A, R = m._design(tau, m._basis(a[k], b[k], dt, x[:n][k], sy), t)
    base = A[:, :, :2] @ th[:2] + A[:, :, 7:] @ th[7:]
    slope = np.median(((y[k] - base) / t)[:, t >= 1.0], axis=1)
    obs_slope = r[k] - R[:, :2] @ th[:2] - R[:, 7:] @ th[7:]
    prior = R[:, 2:7] @ th[2:7]
    error = slope - prior
    print('held-out lab', j, 'n', k.sum(), 'slope bias, median abs, onset-vs-curve median abs:', np.round([np.median(error), np.median(abs(error)), np.median(abs(obs_slope-slope))], 5))
    # Apply the actual training missingness patterns to this held-out lab's complete rows.
    patterns = np.unique(miss[:n][~clean], axis=0)
    if len(patterns):
        med = np.nanmean(np.where(miss[:n][fit], np.nan, x[:n][fit, 1:]), axis=0)
        masked = x[:n][k, None, 1:] - np.where(patterns[None, :, :], x[:n][k, None, 1:] - med, 0.)
        prior_mask = th[2] + masked @ th[3:7]
        print('masked slope median AE, onset fractions 0/.9/1:', np.round([np.median(abs((1-w)*prior_mask+w*obs_slope[:,None]-slope[:,None])) for w in (0., .9, 1.)], 5))
# Verify both polarities and all four yield regimes with an independent state recurrence.
tau = np.array([.62, 4.05]); th = np.array([.66,.56,.08,.15,-.13,.11,-.09,.25,.22])
for sign in (-1., 1.):
    for past_active, hold_active in ((False,False),(True,False),(False,True),(True,True)):
        past = np.full((1,len(dt)), sign*.7*sy)
        if past_active:
            past[0,0] = -sign*1.7*sy
        hold = np.array([sign*(1.3 if hold_active else .6)*sy]); xx = x[:1]
        A,R = m._design(tau,m._basis(past,hold,dt,xx,sy),t)
        z = np.zeros(2); work = 0.
        for h, v in zip(dt, past[0]):
            z = np.exp(-h/tau)*z + (1-np.exp(-h/tau))*v
            work += h*max(abs(v)-sy,0.)**2
        amp = (hold[0]-z)*th[:2]; amp[0] += work*hold[0]*th[8]
        slope = (xx @ th[2:7])[0] + th[7]*sign*max(abs(hold[0])-sy,0.)**2
        assert np.allclose(A @ th, amp @ (1-np.exp(-t/tau[:,None]))+slope*t, atol=1e-12, rtol=0)
        assert np.allclose(R @ th, amp @ (1/tau)+slope, atol=1e-12, rtol=0)
print('Independent signed hold/history recurrence and onset checks passed for all eight cases.')
