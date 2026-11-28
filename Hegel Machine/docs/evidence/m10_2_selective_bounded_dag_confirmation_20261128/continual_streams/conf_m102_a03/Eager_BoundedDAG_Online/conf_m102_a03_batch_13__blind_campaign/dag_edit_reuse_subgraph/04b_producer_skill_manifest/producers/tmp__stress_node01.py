import copy
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
harness = root / 'test_dag_candidate.py'
scope = {'__file__': str(harness), '__name__': '__main__'}
exec(harness.read_text().split('\nst = pickle.load', 1)[0], scope)
s = pickle.loads((root/'public_batch.pkl').read_bytes())[0]['study']
t = s['train']
idx = np.arange(len(t['y']))
folds = [(f'{seed}/{k}', j) for seed in (42,142,242) for k,j in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(idx)),5))]
scores = []
for name,j in folds:
    i = np.setdiff1d(idx,j)
    sample = copy.deepcopy(s)
    sample['train'] = {key: t[key][i] for key in ('x','y','traj')}
    y, end = t['y'][i], t['traj'][i,-1]
    sample['train']['summary'] = np.array([y.mean(), y.std(), end.mean(), end.std()])
    sample['test'] = {'x':t['x'][j]}
    pred, calls = scope['run_dag'](sample)
    assert calls == {'node_00':1,'node_01':1}
    ye = np.abs(pred['y_hat']-t['y'][j]); te = np.abs(pred['state_pred']-t['traj'][j])
    np.testing.assert_allclose(pred['diagnostics_vec'], [pred['y_hat'].mean(),pred['y_hat'].std(),pred['state_pred'][:,-1].mean(),pred['state_pred'][:,-1].std()])
    scores.append([ye.mean(),te.mean(),te.mean(axis=0).max()])
scores = np.array(scores)
print('15 OOF slices, worst scalar / trajectory / individual horizon MAE:', scores.max(axis=0))
assert np.all(scores.max(axis=0) < 0.038)
