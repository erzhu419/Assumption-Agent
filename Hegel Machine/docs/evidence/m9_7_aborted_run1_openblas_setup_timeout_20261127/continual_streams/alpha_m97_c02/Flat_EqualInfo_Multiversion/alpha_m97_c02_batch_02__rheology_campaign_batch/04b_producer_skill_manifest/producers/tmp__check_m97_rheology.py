import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_02_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
past, hold, features, base = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
tr = d['train']
y, r = tr['strain_curves_obs'] - base[:len(tr['lab_id']), None], tr['instant_rate_obs']
lab, strata = tr['lab_id'], tr['stratum']
n = len(lab)
oof_y, oof_r = np.empty_like(y), np.empty_like(r)

def fit(mask):
    return m._fit(past[:n][mask], hold[:n][mask], features[:n][mask], age, dt, t, y[mask], r[mask])

def predict(mask, log_tau, w):
    c, b = m._basis(log_tau, past[:n][mask], hold[:n][mask], features[:n][mask], age, dt, t)
    return c @ w, b @ w

for j in range(4):
    mask = lab == j
    tau, w = fit(~mask)
    oof_y[mask], oof_r[mask] = predict(mask, tau, w)
    print('held lab', j, 'tau', np.round(np.exp(tau), 5), 'weights', np.round(w, 5))

e = oof_y - y
v = oof_r - r
for name, keys in [('lab', lab), ('stratum', strata), ('lab/stratum', lab*10 + strata)]:
    for value in np.unique(keys):
        k = keys == value
        z = e[k]
        print(name, value, 'n', k.sum(), 'curve median abs / p90 / RMSE', np.round([np.median(abs(z)), np.quantile(abs(z), .9), np.sqrt(np.mean(z*z))], 5), 'rate RMSE', round(np.sqrt(np.mean(v[k]**2)), 5))
print('by horizon, median abs', np.round(np.median(abs(e), axis=0), 5))
print('by horizon, count above 0.15', (abs(e) > .15).sum(0))
print('curve abs > .15:', np.round(e[abs(e) > .15], 3), 'rate p90/max abs', np.round(np.quantile(abs(v), [.9, 1]), 5))

out = m.solve_alpha_m97_c02_batch_02(d)
assert np.array_equal(out['regime_mode_weights'], np.zeros(2))
assert np.all(np.diff(out['estimated_taus']) > 0)
# Compute the history step by step independently of the compressed basis.
tau, w = out['estimated_taus'], out['hereditary_weights']
f = past[n:]
z = np.zeros((len(f), 2))
for j, h in enumerate(dt):
    a = np.exp(-h/tau)
    z = a*z + (1-a)*f[:, j, None]
gap = hold[n:, None] - z
drift = features[n:] @ w[2:]
curve = base[n:, None] + (gap*w[:2]) @ (1-np.exp(-t[:, None]/tau)).T + drift[:, None]*t
rate = (gap/tau) @ w[:2] + drift
np.testing.assert_allclose(out['predicted_strain'], curve, atol=1e-12, rtol=0)
np.testing.assert_allclose(out['initial_rate_pred'], rate, atol=1e-12, rtol=0)
print('Full fit', tau, w, '; recurrence and rate agree to 1e-12')
