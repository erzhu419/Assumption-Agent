import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
train = s['train']
idx = np.arange(len(train['x']))
rng = np.random.default_rng(9931)
folds = [('52/20', idx[:52], idx[52:]), ('56/16', idx[:56], idx[56:])]
for i in range(6):
    order = rng.permutation(idx)
    folds.append((f'random-{i}', order[:52], order[52:]))
a = module._reference(s['panel_a'], train['x'])
for j in range(4):
    order = np.argsort(a[:, j])
    for name, tr, te in [('low', order[16:], order[:16]), ('high', order[:-16], order[-16:])]:
        folds.append((f'latent-{j}-{name}', tr, te))
errors = []
for name, tr, te in folds:
    out = module.solve_batch(dict(s, train={k: train[k][tr] for k in ('x', 'y', 'traj')}, test={'x': train['x'][te]}))
    delta = abs(out['state_pred'] - train['traj'][te])
    error = (abs(out['y_hat'] - train['y'][te]).mean(), delta.mean(), delta.mean(axis=0).max())
    errors.append(error)
    print(name, *[f'{e:.6f}' for e in error])
out = module.solve_batch(s)
end = out['state_pred'][:, -1]
y = out['y_hat']
np.testing.assert_allclose(out['diagnostics_vec'], [y.mean(), y.std(), end.mean(), end.std()])
worst = np.max(errors, axis=0)
print('worst scalar, trajectory, horizon:', worst)
assert np.all(worst <= 0.038), worst
