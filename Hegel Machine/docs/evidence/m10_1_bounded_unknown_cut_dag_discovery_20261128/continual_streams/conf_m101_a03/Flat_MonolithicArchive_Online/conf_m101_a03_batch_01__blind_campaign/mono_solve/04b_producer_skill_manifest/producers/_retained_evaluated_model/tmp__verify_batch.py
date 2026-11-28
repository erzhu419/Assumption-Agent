import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('batch_solver', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']
n = len(t['y'])
w = mod._coordinates(s, t['x'])

def checked(study):
    p = mod.solve_batch(study)
    k, nt = len(study['test']['x']), len(study['eval_times'])
    y, z, d = (p[name] for name in ('y_hat', 'state_pred', 'diagnostics_vec'))
    for value, shape in ((y, (k,)), (z, (k, nt)), (d, (4,))):
        assert isinstance(value, np.ndarray) and value.shape == shape and np.isfinite(value).all()
    np.testing.assert_allclose(d, [y.mean(), y.std(), z[:, -1].mean(), z[:, -1].std()], atol=1e-12, rtol=0)
    return y, z

checked(s)

def score(i, j):
    sub = dict(s, train={key: t[key][i] for key in ('x', 'y', 'traj')} | {'summary': t['summary']}, test={'x': t['x'][j]})
    y, z = checked(sub)
    return np.abs(y - t['y'][j]).mean(), np.abs(z - t['traj'][j]).mean(axis=0).max()

rng = np.random.default_rng(179)
groups = {
    'required_52_20': [np.arange(n)],
    'archive_56_16': [np.arange(n)],
    'random_52_20': [rng.permutation(n) for _ in range(24)],
    'directional_52_20': [np.argsort(sign * w[:, k]) for k in range(w.shape[1]) for sign in (-1, 1)],
}
for label, orders in groups.items():
    k = 56 if label == 'archive_56_16' else 52
    scores = np.array([score(q[:k], q[k:]) for q in orders])
    ym, zm = scores.max(axis=0)
    print(f'{label}: worst scalar MAE {ym:.6f}; worst timestamp state MAE {zm:.6f}', flush=True)
    assert ym <= .038, (label, ym)
    assert zm < .008, (label, zm)
