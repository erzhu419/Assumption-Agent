import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m96_w04_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w04_batch_04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
n = len(lab)
x = s._aligned((tr, te))
t = d['eval_times']
y = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
coef = s._calibration(d['calibration_load_cells'])
args = (coef, d['past_dt'], t, d['yield_stress'])

def model(mask):
    return s._Model({k: v[mask] for k, v in tr.items()}, x[:n][mask], *args)

def predict(m, tau, w, missing, rate):
    pred = m.basis(tau)[0] @ w
    use = missing & np.isfinite(rate)
    drift = .97 * (rate[use] - pred[use, -1])
    pred[use, :-1] += drift[:, None] * t
    pred[use, -1] += drift
    return pred

def rms(a):
    return np.sqrt(np.mean(a*a))

# Attribute residual structure using held-out response observations, not calibration-label proxies.
for key, eligible in [('all', np.ones(n, bool)), ('complete_only', complete)]:
    stats = []
    for c in np.unique(lab):
        valid = lab == c
        fit = ~valid & eligible
        tau, w = s._fit(model(fit), y[fit])
        p = predict(model(valid), tau, w, ~complete[valid], y[valid, -1])
        err = p - y[valid]
        m = complete[valid]
        # Orthogonal-to-time residual detects missing memory terms rather than drift imputation.
        curve = err[m, :-1]
        slope = (curve @ t) / (t @ t)
        shape = curve - slope[:, None] * t
        stats.append([rms(err[:, :-1]), rms(curve), rms(err[m, -1]), rms(shape), float(np.mean(slope))])
    print(key, 'held-out lab rows [all_curve, complete_curve, complete_rate, complete_nondrift_shape, signed_drift]', np.round(stats, 6).tolist())
    tau, w = s._fit(model(eligible), y[eligible])
    p = predict(s._Model(te, x[n:], *args), tau, w, ~np.isfinite(te['coupon_features_raw']).all(axis=1), te['instant_rate_obs'])
    e = p[:, :-1] + te['strain_0'][:, None] - g['te_clean']
    r = p[:, -1] - g['te_true_rate0']
    hist = (np.abs(g['te_hist_true']) > d['yield_stress']).any(axis=1)
    hold = np.abs(g['te_hold_true']) > d['yield_stress']
    masks = [(te['lab_id'] == c) & (hist == h) & (hold == j) for c in range(4) for h in (0, 1) for j in (0, 1)]
    print(key, 'public worst lab-by-yield', max(rms(e[m]) for m in masks if m.any()), max(rms(r[m]) for m in masks if m.any()), 'tau max rel', np.max(np.abs(tau/g['taus']-1)), 'weight max abs', np.max(np.abs(w-np.r_[g['theta'],g['vp_theta']])))

# Independent loading recursion and explicit right derivative verify the contract's algebra.
m = model(np.ones(n, bool))
tau, w = s._fit(m, y)
B = m.basis(tau)[0]
for j, tj in enumerate(tau):
    z = np.zeros(n)
    for k, step in enumerate(d['past_dt']):
        z += (m.past[:, k] - z) * -np.expm1(-step / tj)
    np.testing.assert_allclose(B[:, :-1, j], (m.hold-z)[:, None] * -np.expm1(-t/tj), atol=1e-12)
    np.testing.assert_allclose(B[:, -1, j], (m.hold-z)/tj, atol=1e-12)
np.testing.assert_allclose(B[:, -1, 7], m.over, atol=1e-12)
np.testing.assert_allclose(B[:, -1, 8], m.work*m.hold/tau[0], atol=1e-12)
print('Independent recurrence and right derivatives agree')
