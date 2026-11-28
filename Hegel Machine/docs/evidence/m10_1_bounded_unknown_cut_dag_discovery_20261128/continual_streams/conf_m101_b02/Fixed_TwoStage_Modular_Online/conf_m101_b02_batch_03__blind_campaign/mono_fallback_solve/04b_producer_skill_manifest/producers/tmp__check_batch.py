import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
n = len(tr['x'])
idx = np.arange(n)
folds = [('fixed52', idx[:52], idx[52:])]
for seed in range(10, 30):
    p = np.random.default_rng(seed).permutation(n)
    folds.append((f'random{seed}', p[:56], p[56:]))
r = np.linalg.norm(tr['x'], axis=1)
p = np.argsort(r)
folds.append(('extreme_norm', p[:-12], p[-12:]))
errors = []
for name, a, b in folds:
    split = dict(s, train=dict(tr, **{k:tr[k][a] for k in ('x','y','traj')}), test={'x':tr['x'][b]})
    out = m.solve_batch(split)
    dy = abs(out['y_hat'] - tr['y'][b])
    dt = abs(out['state_pred'] - tr['traj'][b])
    e = np.r_[dy.mean(), dt.mean(), dt.mean(0)]
    assert np.isfinite(e).all()
    errors.append(e)
    if name in ('fixed52', 'extreme_norm'):
        print(name, np.round(e, 5), flush=True)
errors = np.array(errors)
print('random56 worst per metric', np.round(errors[1:-1].max(0), 5))
print('random56 worst y fold', folds[1+errors[1:-1, 0].argmax()][0])
assert errors[0, 0] <= .038
