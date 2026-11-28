import copy, importlib.util, pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

root = '/holdout_task/'
zero = load('zero', root + '_frozen_nodes/node_00.py')
one = load('one', root + 'node_01.py')
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
x = zero.run_node({'__input__': s})['x_tr']
b = one._panel_coordinates(s['panel_b'], x)
y, z = s['train']['y'], s['train']['traj']
ix = np.arange(len(x))
folds = [*np.array_split(ix, 4), np.argsort(np.max(abs(x), axis=1))[-16:], np.argsort(np.linalg.norm(b, axis=1))[-16:]]
metrics = []
for te in folds:
    tr = np.setdiff1d(ix, te)
    part = copy.deepcopy(s)
    part['train'] = {k: s['train'][k][tr] for k in ('x', 'y', 'traj')}
    part['train']['summary'] = np.array([y[tr].mean(), y[tr].std(), z[tr,-1].mean(), z[tr,-1].std()])
    part['test'] = {'x': s['train']['x'][te]}
    out = one.run_node({'__input__': part, 'node_00': zero.run_node({'__input__': part})})
    trajectory = abs(out['state_pred'] - z[te])
    metrics.append((np.mean(abs(out['y_hat'] - y[te])), trajectory.mean(), trajectory.mean(axis=0).max()))
m = np.array(metrics)
print('End-to-end slices (scalar, trajectory, worst horizon MAE):', m, flush=True)
assert np.isfinite(m).all() and m[:4, 0].max() < .038 and np.all(m[4:, 0] < [.057402, .058667]) and m[:, 2].max() < .038
