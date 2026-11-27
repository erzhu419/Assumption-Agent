import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

path = '/holdout_task/alpha_m96_w01_s00_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m96_w01_s00.pkl', 'rb') as f:
    case = pickle.load(f)[0]
d, gt = case['dataset'], case['gt']
tr, te = d['train'], d['test']
x, missing = m._features(tr, te)
lab = tr['lab_id']
n = len(lab)
a, b = m._load_calibration(d['calibration_load_cells'])
dt, t = d['past_dt'], d['eval_times']

def stresses(split):
    j = split['lab_id']
    return split['past_stress_cmd'] * a[j, None] + b[j, None], split['stress_hold_cmd'] * a[j] + b[j]

out = m.solve_alpha_m96_w01_s00(d)
tau, theta = out['estimated_taus'], out['hereditary_weights']
past, hold = stresses(te)
z = np.zeros((len(hold), 2))
for k, interval in enumerate(dt):
    decay = np.exp(-interval / tau)
    z = z * decay + past[:, k, None] * (1 - decay)
amp = (hold[:, None] - z) * theta[:2]
slope = x[n:] @ theta[2:]
update = missing[n:].any(1) & np.isfinite(te['instant_rate_obs'])
slope[update] = te['instant_rate_obs'][update] - np.sum(amp[update] / tau, axis=1)
explicit_curve = te['strain_0'][:, None] + amp @ (-np.expm1(-t[None, :] / tau[:, None])) + slope[:, None] * t
explicit_rate = np.sum(amp / tau, axis=1) + slope
for key, ref in [('predicted_strain', explicit_curve), ('initial_rate_pred', explicit_rate)]:
    err = np.max(np.abs(out[key] - ref))
    print(key, 'explicit recurrence/derivative max discrepancy', err)
    assert err < 1e-11
assert .20 <= tau[0] < tau[1] <= 8 and np.array_equal(out['viscoplastic_weights'], np.zeros(2))
print('public total and worst-horizon strain RMSE', np.sqrt(np.mean((explicit_curve-gt['te_clean'])**2)), np.sqrt(np.max(np.mean((explicit_curve-gt['te_clean'])**2, axis=0))))

# Fit challengers only on each fold's training rows. Audit the complete-coupon
# prior so the rate-assisted prediction cannot mask model misspecification.
past, hold = stresses(tr)
x = x[:n]
complete = ~missing[:n].any(1)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
rate = tr['instant_rate_obs']
fold = np.full(n, -1)
rng = np.random.default_rng(12)
for j in np.unique(lab):
    ids = rng.permutation(np.flatnonzero(complete & (lab == j)))
    fold[ids] = np.arange(len(ids)) % 4
pair = np.column_stack([x[:, i] + sign*x[:, j] for i in range(1,5) for j in range(i+1,5) for sign in (-1, 1)])
nonlinear = np.column_stack((pair**2/(1 + pair**2), pair/(1 + pair**2), x[:,1]*x[:,2]*x[:,3], x[:,2]*x[:,3]*x[:,4]))
lab_terms = np.eye(4)[lab][:,1:]
err = {key: np.full_like(y, np.nan) for key in ('base', 'lab', 'nonlinear')}
rate_err = {key: np.full(n, np.nan) for key in err}
for k in range(4):
    fit = complete & (fold != k)
    val = fold == k
    taus, w = m._fit(past[fit], hold[fit], dt, t, x[fit], y[fit], rate[fit])
    h, r = m._design(taus, past, hold, dt, t, x)
    for key, f in [('base', np.empty((n,0))), ('lab', lab_terms), ('nonlinear', nonlinear)]:
        cm = fit[:, None] & np.isfinite(y)
        rm = fit & np.isfinite(rate)
        if f.shape[1]:
            hc = np.concatenate((h, f[:,None,:]*t[None,:,None]), axis=2)
            rc = np.column_stack((r, f))
            v = least_squares(lambda p: np.r_[hc[cm] @ p-y[cm], rc[rm] @ p-rate[rm]], np.r_[w,np.zeros(f.shape[1])], loss='cauchy', f_scale=.025).x
        else:
            hc, rc, v = h, r, w
        err[key][val] = hc[val] @ v - y[val]
        rate_err[key][val] = rc[val] @ v - rate[val]
for key in err:
    e, r = err[key], rate_err[key]
    print(key, 'out-of-fold curve median absolute errors by lab', [round(float(np.nanmedian(abs(e[lab==j]))),6) for j in range(4)], 'rate RMSE by lab', [round(float(np.sqrt(np.nanmean(r[lab==j]**2))),6) for j in range(4)], 'median signed curve errors by lab', [round(float(np.nanmedian(e[lab==j])),6) for j in range(4)])
