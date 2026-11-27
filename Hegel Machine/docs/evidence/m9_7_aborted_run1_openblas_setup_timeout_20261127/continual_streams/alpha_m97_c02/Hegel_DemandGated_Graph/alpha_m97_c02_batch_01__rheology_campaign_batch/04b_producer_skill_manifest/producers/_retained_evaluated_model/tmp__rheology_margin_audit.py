import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
p, h, x, b = m._prepare(d)
tr = d['train']; labs = tr['lab_id']; n = len(labs)
dt, t = d['past_dt'], d['eval_times']
a = m._activity(d, p, dt)
y = tr['strain_curves_obs'] - b[:n, None]; r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('Incomplete training rows per lab:', [(c, int(sum((labs == c) & ~complete))) for c in range(4)])

out = m.solve_alpha_m97_c02_batch_01(d)
tau = out['estimated_taus']; w = out['hereditary_weights']; v = out['regime_mode_weights']
z = np.zeros((len(p), 2))
work = np.zeros(len(p))
for j, step in enumerate(dt):
    decay = np.exp(-step / tau)
    z = decay * z + (1 - decay) * p[:, j, None]
    work += np.maximum(0, np.abs(p[:, j]) - d['crit_stress'])**2 * step
np.testing.assert_allclose(a, np.tanh(work), atol=1e-13)
gap = h[:, None] - z
rise = 1 - np.exp(-t[:, None] / tau)
for plateau in (False, True):
    activity = a if plateau else np.zeros_like(a)
    drift = x @ w[2:] + v[1] * activity * z[:, 0]
    amp = gap * w[:2]; amp[:, 0] += v[0] * activity * h
    expected_curve = amp @ rise.T + drift[:, None] * (np.log1p(t) if plateau else t)
    expected_rate = amp @ (1 / tau) + drift
    A, R = m._basis(tau, p, h, x, dt, t, activity, plateau)
    coef = np.r_[w, v] if plateau else w
    np.testing.assert_allclose(A @ coef, expected_curve, atol=1e-12)
    np.testing.assert_allclose(R @ coef, expected_rate, atol=1e-12)
    if plateau:
        np.testing.assert_allclose(out['predicted_strain'], b[n:, None] + expected_curve[n:], atol=1e-12)
        np.testing.assert_allclose(out['initial_rate_pred'], expected_rate[n:], atol=1e-12)
print('Independent recurrence, both flow laws, onset derivative and output handoff passed.')

for label, eligible in [('all', np.ones(n, dtype=bool)), ('complete-only', complete)]:
    worst = np.zeros(3)
    for c in range(4):
        fit = (labs != c) & eligible
        test = (labs == c) & complete
        ta, co = m._fit(p[:n][fit], h[:n][fit], x[:n][fit], dt, t, a[:n][fit], y[fit], r[fit])
        A, R = m._basis(ta, p[:n][test], h[:n][test], x[:n][test], dt, t, a[:n][test])
        ec, er = A @ co - y[test], R @ co - r[test]
        metric = np.array([np.median(np.abs(ec)), np.median(np.abs(er)), abs(np.median(ec[:, -1]))])
        worst = np.maximum(worst, metric)
        print(label, 'held lab', c, 'tau', np.round(ta, 5), 'curve/rate MedAE and terminal bias', np.round(metric, 6))
    print(label, 'worst held-lab metrics:', np.round(worst, 6))
    ta, co = m._fit(p[:n][eligible], h[:n][eligible], x[:n][eligible], dt, t, a[:n][eligible], y[eligible], r[eligible])
    A, R = m._basis(ta, p[n:], h[n:], x[n:], dt, t, a[n:])
    ec, er = b[n:, None] + A @ co - gt['te_clean'], R @ co - gt['te_true_rate0']
    rmse = lambda v: np.sqrt(np.mean(v*v))
    print(label, 'public relative tau max, curve RMSE, rate RMSE:', max(abs(ta / gt['taus'] - 1)), rmse(ec), rmse(er))
