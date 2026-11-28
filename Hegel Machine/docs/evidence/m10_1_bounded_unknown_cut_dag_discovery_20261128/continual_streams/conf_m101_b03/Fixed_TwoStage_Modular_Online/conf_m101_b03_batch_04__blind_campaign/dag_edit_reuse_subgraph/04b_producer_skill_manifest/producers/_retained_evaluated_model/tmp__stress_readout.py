import runpy
import pickle
import numpy as np
base = '/holdout_task/'
calibrate = runpy.run_path(base + '_frozen_nodes/node_00.py')['run_node']
run = runpy.run_path(base + 'node_01.py')['run_node']
with open(base + 'public_batch.pkl', 'rb') as file:
    study = pickle.load(file)[0]['study']
d = calibrate({'__input__': study})
n = d['n_train']; x = d['z'][:n]; y = d['y']; traj = d['traj']
rad = np.linalg.norm(x, axis=1)
slices = {'tail': [np.arange(n-20, n)],
          'random': [j for seed in (39, 101, 211) for j in np.array_split(np.random.default_rng(seed).permutation(n), 6)],
          'outer-radius': [np.argsort(rad)[-12:]]}
for name, folds in slices.items():
    errors = []
    for j in folds:
        i = np.setdiff1d(np.arange(n), j)
        out = run({'node_00': dict(d, z=x[np.r_[i,j]], n_train=len(i), y=y[i], traj=traj[i])})
        delta = np.abs(out['state_pred'] - traj[j]); p = out['y_hat']; q = out['state_pred'][:, -1]
        np.testing.assert_allclose(out['diagnostics_vec'], [p.mean(), p.std(), q.mean(), q.std()])
        errors.append([np.mean(np.abs(p-y[j])), delta.mean(), *delta.mean(0)])
    print(name, 'worst response / aggregate trajectory / each horizon:', np.max(errors, axis=0).round(6))
