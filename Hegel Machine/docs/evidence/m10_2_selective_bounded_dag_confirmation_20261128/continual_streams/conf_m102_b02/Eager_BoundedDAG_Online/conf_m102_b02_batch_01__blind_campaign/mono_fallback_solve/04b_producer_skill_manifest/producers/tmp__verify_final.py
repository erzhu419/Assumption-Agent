import importlib.util
import pickle
import numpy as np
p = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(p)
p.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
n = len(tr['x'])
z, _ = m._coordinates(s, tr['x'])
rng = np.random.default_rng(20531)
folds = [np.arange(52, n)] + [rng.choice(n, 20, replace=False) for _ in range(40)] + [np.argsort(np.linalg.norm(z, axis=1))[-20:]]
metrics = []
for j in folds:
    i = np.setdiff1d(np.arange(n), j)
    train = dict(tr, **{k:tr[k][i] for k in ('x','y','traj')})
    out = m.solve_batch(dict(s,train=train,test={'x':tr['x'][j]}))
    e = abs(out['state_pred'] - tr['traj'][j])
    metrics.append([abs(out['y_hat']-tr['y'][j]).mean(), e.mean(), e[:,-1].mean(), e.mean(0).max()])
a = np.array(metrics)
assert np.isfinite(a).all() and (a[0,:2] <= .038).all()
print('52/20:', a[0], 'shuffle worst:', a[1:-1].max(0), 'large radius:', a[-1])
