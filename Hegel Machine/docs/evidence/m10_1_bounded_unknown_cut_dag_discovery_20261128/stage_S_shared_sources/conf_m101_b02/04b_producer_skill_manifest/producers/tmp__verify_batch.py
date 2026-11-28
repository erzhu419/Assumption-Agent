import importlib.util
import pickle
import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x = t['x']; y = t['y']; z = t['traj']
idx = np.arange(len(x)); rng = np.random.RandomState(228)
r = np.argsort(np.linalg.norm(x, axis=1))[-15:]
folds = [('prescribed', idx[52:]), *[(f'fold{i}', j) for i, j in enumerate(np.array_split(rng.permutation(len(x)), 5))], ('high-amplitude', r)]
for path in ('/tmp/batch_solver_v01.py', '/holdout_task/batch_solver.py'):
    solve = load(path).solve_batch
    results = []
    for name, va in folds:
        tr = np.setdiff1d(idx, va)
        p = solve(dict(s, train=dict(t, x=x[tr], y=y[tr], traj=z[tr]), test={'x': x[va]}))
        e = abs(p['state_pred'] - z[va]); m = np.r_[abs(p['y_hat'] - y[va]).mean(), e.mean(), e.mean(0)[-1], e.mean(0).max()]
        results.append(m)
        print(path, name, np.round(m, 6))
    p = solve(s); a = p['y_hat']; b = p['state_pred']; d = p['diagnostics_vec']
    assert a.shape == (36,) and b.shape == (36, 8) and d.shape == (4,)
    assert all(np.isfinite(q).all() for q in (a,b,d))
    np.testing.assert_allclose(d, [a.mean(), a.std(), b[:, -1].mean(), b[:, -1].std()])
    assert results[0][0] <= .038
    print('worst', np.round(np.max(results, axis=0), 6))
