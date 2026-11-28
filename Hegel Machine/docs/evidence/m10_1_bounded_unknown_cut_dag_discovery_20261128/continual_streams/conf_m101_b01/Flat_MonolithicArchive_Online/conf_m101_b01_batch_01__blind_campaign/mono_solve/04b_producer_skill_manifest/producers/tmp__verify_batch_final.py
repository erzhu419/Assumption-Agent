import importlib.util
import pickle
import numpy as np
from scipy.spatial.distance import cdist

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']
x = m._calibrate(s['panel_a'], t['x'])
n = len(x)
idx = np.arange(n)

def measure(i, j):
    yy, tt = t['y'][i], t['traj'][i]
    a = dict(s, train=dict(x=t['x'][i], y=yy, traj=tt,
                          summary=np.array([yy.mean(), yy.std(), tt[:, -1].mean(), tt[:, -1].std()])), test={'x': t['x'][j]})
    o = m.solve_batch(a)
    y, tr = o['y_hat'], o['state_pred']
    assert np.allclose(o['diagnostics_vec'], [y.mean(), y.std(), tr[:, -1].mean(), tr[:, -1].std()])
    ey, et = abs(y-t['y'][j]), abs(tr-t['traj'][j])
    return np.r_[ey.mean(), et.mean(), et.mean(axis=0).max()]

for k in (52, 56):
    print(f'{k}/{n-k} y, trajectory, worst horizon MAE:', np.round(measure(idx[:k], idx[k:]), 7))
folds = np.array_split(np.random.RandomState(91).permutation(n), 4)
e = np.array([measure(np.setdiff1d(idx, j), j) for j in folds])
print('Randomized fold maxima:', np.round(e.max(0), 7))
near = np.partition(cdist(x, x), 5, axis=1)[:, 5]
boundary = [np.argsort(v) for v in (near, np.linalg.norm(x, axis=1), *x.T, *(-x).T)]
e = np.array([measure(j[:-15], j[-15:]) for j in boundary])
print('Boundary slice maxima:', np.round(e.max(0), 7))
assert np.max(e[:, :2]) <= 0.038
