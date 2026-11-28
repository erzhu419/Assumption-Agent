import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']
a = s['train']; x = a['x']; all_idx = np.arange(len(x)); rng = np.random.default_rng(413)
scale = np.linalg.norm((x-x.mean(0))/x.std(0), axis=1)
slices = [('prescribed', all_idx[52:]), ('outer-radius', np.argsort(scale)[-20:])]
for j in range(4):
    order = np.argsort(x[:, j]); slices += [(f'x{j}-low', order[:20]), (f'x{j}-high', order[-20:])]
slices += [(f'random-{j}', rng.permutation(len(x))[:20]) for j in range(10)]
worst = np.zeros(3)
for name, hold in slices:
    fit = np.setdiff1d(all_idx, hold)
    ss = dict(s, train={k: a[k][fit] for k in ('x', 'y', 'traj')}, test={'x': x[hold]})
    out = m.solve_batch(ss)
    ey = np.abs(out['y_hat'] - a['y'][hold]); et = np.abs(out['state_pred'] - a['traj'][hold]); score = np.array([ey.mean(), et.mean(), et.mean(0).max()]); worst = np.maximum(worst, score)
    print(name, np.round(score, 7))
    assert np.all(score <= .038)
out = m.solve_batch(s)
assert out['y_hat'].shape == (36,) and out['state_pred'].shape == (36, 8) and out['diagnostics_vec'].shape == (4,)
assert all(np.isfinite(v).all() for v in out.values())
y, terminal = out['y_hat'], out['state_pred'][:, -1]
np.testing.assert_array_equal(out['diagnostics_vec'], [y.mean(), y.std(), terminal.mean(), terminal.std()])
print('worst (y, trajectory, horizon):', worst)
