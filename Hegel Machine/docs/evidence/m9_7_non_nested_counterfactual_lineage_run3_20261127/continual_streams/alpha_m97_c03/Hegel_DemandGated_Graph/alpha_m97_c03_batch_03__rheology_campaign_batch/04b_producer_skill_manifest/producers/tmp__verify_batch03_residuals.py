import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb'))[0]['dataset']
train = d['train']
n = len(train['lab_id'])
lab = np.asarray(train['lab_id'])
stratum = np.asarray(train['stratum'])
missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)
y, rate = train['strain_curves_obs'], train['instant_rate_obs']
past, hold, drift, baseline = m._prepare(d)
dt, times = d['past_dt'], d['eval_times']
full = m.solve_alpha_m97_c03_batch_03(d)
tau = full['estimated_taus']
w = np.r_[full['hereditary_weights'], full['regime_mode_weights']]
print('full taus', tau, 'weights', w)
assert np.all(tau > 0)
z = np.zeros((len(past), 2))
for j, step in enumerate(dt):
    a = np.exp(-step / tau)
    z = a * z + (1-a) * past[:, j, None]
g0, g1 = (hold[:, None] - z).T
v = w[2] + drift[:, 1:] @ w[3:7] + w[7] * np.sign(hold) * np.sqrt(np.abs(hold))
ref_rate = (w[0]*g0 + w[8]*(g0-g1))/tau[0] + w[1]*g1/tau[1] + v
ref_curve = baseline[:, None] + (w[0]*g0 + w[8]*(g0-g1))[:, None]*(-np.expm1(-times/tau[0])) + (w[1]*g1)[:, None]*(-np.expm1(-times/tau[1])) + v[:, None]*times/(1+.85*times)
np.testing.assert_allclose(full['predicted_strain'], ref_curve[n:], atol=1e-12)
np.testing.assert_allclose(full['initial_rate_pred'], ref_rate[n:], atol=1e-12)
check_curve, _ = m._design(tau, past[n:], hold[n:], drift[n:], dt, np.array([0., 1e-7]))
np.testing.assert_allclose(check_curve[:, 0] @ w, 0., atol=1e-12)
np.testing.assert_allclose(((check_curve[:, 1]-check_curve[:, 0]) @ w)/1e-7, ref_rate[n:], atol=1e-5, rtol=1e-5)

def error(x):
    return 'rmse %.6g, medAE %.6g, q90AE %.6g' % (np.sqrt(np.nanmean(x*x)), np.nanmedian(np.abs(x)), np.nanquantile(np.abs(x), .9))

for group, labels in [('lab', lab), ('stratum', stratum)]:
    ec, er = np.empty_like(y), np.empty_like(rate)
    for label in np.unique(labels):
        val = labels == label
        fit, pred = np.flatnonzero(~val), np.flatnonzero(val)
        t, coef = m._fit(past[fit], hold[fit], drift[fit], dt, times, y[fit]-baseline[fit,None], rate[fit])
        a, b = m._design(t, past[pred], hold[pred], drift[pred], dt, times)
        ec[val], er[val] = baseline[pred,None] + a @ coef - y[val], b @ coef - rate[val]
        print(group, label, 'taus', t, 'regime', coef[7:], 'curve', error(ec[val]), 'rate', error(er[val]))
    print(group, 'worst horizon', np.argmax(np.nanmean(ec*ec, axis=0)), error(ec[:, np.argmax(np.nanmean(ec*ec, axis=0))]))
    for name, mask in [('missing coupon', missing), ('complete coupon', ~missing), ('negative hold', hold[:n] < 0), ('nonnegative hold', hold[:n] >= 0)]:
        print(group, name, 'n', mask.sum(), 'curve', error(ec[mask]), 'rate', error(er[mask]))

    if group == 'lab':
        ranked = np.argsort(np.nan_to_num(np.abs(ec), nan=-1).ravel())[::-1][:6]
        for row, col in zip(*np.unravel_index(ranked, ec.shape)):
            print('largest strain residual', 'lab', lab[row], 'stratum', stratum[row], 'horizon', col, 'observed', round(y[row,col], 4), 'predicted', round(y[row,col]+ec[row,col], 4), 'same-row residuals', np.round(ec[row], 3), 'rate residual', round(er[row], 4))
