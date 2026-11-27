import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch06', '/holdout_task/alpha_m97_c01_batch_06_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_06.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
lab, stratum = tr['lab_id'], tr['stratum']
n = len(lab)
past, hold, x, base = s._inputs(d)
out = s.solve_alpha_m97_c01_batch_06(d)
tau = out['estimated_taus']
hw, rw = out['hereditary_weights'], out['regime_mode_weights']
z = np.zeros((len(hold)-n, 2))
for j, dt in enumerate(d['past_dt']):
    a = np.exp(-dt/tau)
    z = a*z + (1-a)*past[n:, j, None]
g0, g1 = (hold[n:, None] - z).T
v = hw[2] + x[n:] @ hw[3:] + rw[0]*np.sign(hold[n:])*np.sqrt(abs(hold[n:]))
def curve(t):
    t = np.atleast_1d(t)[None, :]
    return (base[n:, None] + hw[0]*g0[:, None]*(-np.expm1(-t/tau[0]))
            + hw[1]*g1[:, None]*(-np.expm1(-t/tau[1]))
            + rw[1]*(g0-g1)[:, None]*(-np.expm1(-t/tau[0]))
            + v[:, None]*t/(1+0.85*t))
curve_error = np.max(abs(curve(d['eval_times'])-out['predicted_strain']))
h = 1e-5
rate_error = np.max(abs(((curve(h)-curve(-h))/(2*h))[:, 0]-out['initial_rate_pred']))
print('independent rational/slip/exchange curve max discrepancy', curve_error, 'numerical initial-rate max discrepancy', rate_error)
assert curve_error < 1e-11 and rate_error < 1e-8
c = d['calibration_load_cells']
a, b = s._calibrate(d)[c['lab_id']].T
for ell in range(4):
    residual = abs(c['true_stress'] - a*c['cmd_stress'] - b)[c['lab_id']==ell]
    clean = residual <= 3
    print('calibration lab', ell, 'clean count', clean.sum(), 'max clean residual', residual[clean].max(), 'min EMI residual', residual[~clean].min() if (~clean).any() else None)
    assert clean.any() and residual[clean].max() < 0.001
model = s._Model(d, past, hold, x)
y = np.c_[tr['strain_curves_obs']-base[:n,None], tr['instant_rate_obs']]
complete = np.isfinite(tr['coupon_features_raw']).all(1)
stats = []
for ell in np.unique(lab):
    t, w = s._fit(model, y, np.flatnonzero(complete & (lab!=ell)))
    ix = np.flatnonzero(lab==ell)
    e = model.design(t, ix) @ w - y[ix]
    for group in np.unique(stratum[ix]):
        m = stratum[ix]==group
        ec = e[m & complete[ix]]
        assert len(ec)
        stats.append([abs(np.median(ec[:,:-1])), abs(np.median(ec[:,-1])), np.median(abs(ec[:,:-1])), np.median(abs(ec[:,-1]))])
        print('held-out lab/stratum', ell, group, 'n complete', len(ec), 'absolute median signed curve/rate', np.round(stats[-1][:2], 6), 'median absolute curve/rate', np.round(stats[-1][2:], 6))
print('worst cell', np.max(stats, axis=0))
