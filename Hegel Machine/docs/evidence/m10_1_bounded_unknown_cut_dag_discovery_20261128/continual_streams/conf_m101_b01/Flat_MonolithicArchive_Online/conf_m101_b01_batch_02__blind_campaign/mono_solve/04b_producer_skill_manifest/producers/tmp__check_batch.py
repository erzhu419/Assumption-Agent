import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x = t['x']; y = t['y']; tr = t['traj']; ix = np.arange(len(x))
z, q = m._coordinates(s, x)
order = np.argsort(np.linalg.norm(z, axis=1))
folds = [('required 52/20', ix[:52], ix[52:]), ('archive 56/16', ix[:56], ix[56:]), ('large input', order[:56], order[56:])]
for seed in (234, 5247):
    for k, b in enumerate(np.array_split(np.random.default_rng(seed).permutation(ix), 5)):
        folds.append((f'{seed}/{k}', np.setdiff1d(ix, b), b))
metrics = []
for name, a, b in folds:
    st = dict(s, train={**t, **{key: t[key][a] for key in ('x', 'y', 'traj')}}, test={'x': x[b]})
    out = m.solve_batch(st)
    et = abs(out['state_pred'] - tr[b]); ey = abs(out['y_hat'] - y[b])
    d = m._affine(q)
    direct = abs(d[b] @ m._fit(d[a], tr[a]) - tr[b])
    vec = np.array([ey.mean(), et.mean(), et.mean(axis=0).max(), et[:, -1].mean(), direct.mean()])
    metrics.append(vec)
    print(name, 'y/state/worst-time/terminal/direct', np.round(vec, 7))
assert np.max(metrics, axis=0)[0] <= 0.038
assert np.max(metrics, axis=0)[1] <= 0.038
print('worst across slices:', np.max(metrics, axis=0))
