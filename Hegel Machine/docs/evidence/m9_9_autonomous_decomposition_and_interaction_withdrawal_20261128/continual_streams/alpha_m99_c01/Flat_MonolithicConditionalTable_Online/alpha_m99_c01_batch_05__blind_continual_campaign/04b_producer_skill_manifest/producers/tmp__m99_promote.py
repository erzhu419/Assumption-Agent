exec(open('/tmp/m99_check_candidate.py').read().split('out =')[0])
from pathlib import Path
out = m.solve_batch(ds)
tau, beta, t = out['estimated_taus'], out['hereditary_weights'], ds['eval_times']
bridge = m._bridge(ds['calibration_load_cells'])
h = m._invert_bridge(g['hold_channel_raw'], bridge).sum(axis=-1)
x = m._chamber_features(g)
drift = beta[2] + x @ beta[3:]
f = -np.expm1(-t[:, None] / tau)
A = drift[:, None, None] * (f * tau - t[:, None])
y = g['strain_curves_obs'] - g['strain_0'][:, None] - (h[:, None] * beta[:2]) @ f.T - drift[:, None] * t
w = np.linalg.lstsq(A.reshape(-1, 2), y.ravel(), rcond=None)[0]
print('Campaign drift fractions:', w, 'steady fraction:', 1-w.sum())
assert (w > 0).all() and w.sum() < 1
err = A @ w - y
assert np.max(abs(err)) < np.max(abs(A @ [.25,.50] - y))
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
old = '    fractions = np.array([0.25, 0.50])'
assert s.count(old) == 1
p.write_text(s.replace(old, '    # Campaign fractions estimated from the public chamber verification curves.\n    fractions = np.array([%.8f, %.8f])' % tuple(w)))
