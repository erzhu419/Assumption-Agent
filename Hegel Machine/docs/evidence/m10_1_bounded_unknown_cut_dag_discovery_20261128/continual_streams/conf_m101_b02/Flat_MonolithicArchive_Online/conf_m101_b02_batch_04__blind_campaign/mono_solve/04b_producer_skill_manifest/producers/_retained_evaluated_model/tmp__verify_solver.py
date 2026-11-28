import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('batch_solver', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
train = study['train']
indices = np.arange(len(train['x']))
a = (train['x'] - mod._reference(study['panel_a'])[0]) / mod._reference(study['panel_a'])[1]
rng = np.random.default_rng(87)
slices = {'52/20': indices[52:], '56/16': indices[56:]}
slices.update({f'fold {k+1}': v for k, v in enumerate(np.array_split(rng.permutation(len(indices)), 5))})
slices['large norm'] = np.argsort(np.linalg.norm(a, axis=1))[-12:]
slices['large coordinate'] = np.argsort(np.max(np.abs(a), axis=1))[-12:]

def check(test_study):
    result = mod.solve_batch(test_study)
    y, state, diag = (result[k] for k in ('y_hat', 'state_pred', 'diagnostics_vec'))
    n = len(test_study['test']['x'])
    assert (y.shape, state.shape, diag.shape) == ((n,), (n, len(study['eval_times'])), (4,))
    assert all(np.isfinite(v).all() for v in (y, state, diag))
    np.testing.assert_allclose(diag, [y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return y, state

check(study)
metrics = []
for name, val in slices.items():
    it = np.setdiff1d(indices, val)
    data = {k: train[k][it] for k in ('x', 'y', 'traj')}
    last = data['traj'][:, -1]
    data['summary'] = np.array([data['y'].mean(), data['y'].std(), last.mean(), last.std()])
    y, state = check(dict(study, train=data, test={'x': train['x'][val]}))
    err = np.abs(state - train['traj'][val])
    metric = np.array([np.mean(np.abs(y - train['y'][val])), err.mean(), err.mean(0).max()])
    print(name, *[f'{v:.6f}' for v in metric])
    metrics.append(metric)
worst = np.max(metrics, axis=0)
print('worst (scalar, trajectory, timestamp):', *[f'{v:.6f}' for v in worst])
assert np.all(worst < 0.038)
