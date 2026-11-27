import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
item = pickle.load(open(root + 'public_alpha_m97_c02_s00.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
cal = s._calibrate(d['calibration_load_cells'])
print('public max gain/bias error', np.max(np.abs(cal - np.column_stack((gt['stress_gains'], gt['stress_biases']))), axis=0))
print('reference shapes', tr['ref_coupons'].shape, te['ref_coupons'].shape)
# Rescale real stress experiments by channel and inject one calibration-only spike.
# This tests whether affine correction and clean in-batch precedence survive an EMI excursion.
g = np.array([.83, 1.18, .91, 1.25]); b = np.array([.21, -.17, .37, -.24])
rows = d['calibration_load_cells']; ls = rows['lab_id']
synthetic = dict(rows, cmd_stress=(rows['cmd_stress']-b[ls])/g[ls], true_stress=rows['true_stress'].copy())
synthetic['true_stress'][np.flatnonzero(ls == 2)[3]] += 5.0
expected = np.column_stack((cal[:, 0] * g, cal[:, 1] + cal[:, 0] * b))
err = np.abs(s._calibrate(synthetic) - expected)
print('transformed calibration with EMI max gain/bias error', err.max(axis=0))
assert err.max() < .001
n = len(tr['lab_id']); lab = tr['lab_id']; x = s._align(tr, te)[:n]
past, hold = s._physical(tr, cal); dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]; r = tr['instant_rate_obs']
miss = np.isnan(tr['coupon_features_raw']).any(axis=1)
# Compare the prescribed linear envelope with a quadratic time term. Fit the
# extra term inside each outer fold on complete coupons only, using robust updates.
for l in np.unique(lab):
    v = lab == l; q = ~v
    tau, w = s._fit(past[q], hold[q], x[q], dt, t, y[q], r[q])
    a, rate = s._basis(past, hold, dt, t, tau)
    drift = w[2] + x @ w[3:]; e = y - (a @ w[:2] + drift[:, None]*t)
    er = r - (rate @ w[:2] + drift)
    # Differences of (strain minus memory)/t cancel the entire linear drift.
    j = np.array([1, 2, 3]); k = np.array([-4, -3, -2])
    curvature = np.median((e[:, k]/t[k] - e[:, j]/t[j]) / (t[k]-t[j]), axis=1)
    c = np.median(curvature[q & ~miss]); clean = v & ~miss
    print('lab', int(l), 'quadratic train/validation coefficients', round(c,6), round(np.median(curvature[clean]),6), 'complete median late-strain baseline/challenger', *[round(np.median(np.abs(z[clean,-2:])),6) for z in (e, e-c*t*t)], 'signed complete rate median', round(np.median(er[clean]),6))
