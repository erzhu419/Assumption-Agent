import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
train = study['train']
x = train['x']
y = train['y']
T = train['traj']
ids = np.arange(len(x))
r = np.linalg.norm(x, axis=1)
axes = [np.argsort(abs(x[:, j]))[-14:] for j in range(x.shape[1])]
folds = [ids[56:], *np.array_split(ids, 4), np.argsort(r)[-14:], *axes]
errors = []
for jj in folds:
    ii = np.setdiff1d(ids, jj)
    q = {**study, 'train': {**train, 'x': x[ii], 'y': y[ii], 'traj': T[ii]}, 'test': {'x': x[jj]}}
    out = mod.solve_batch(q)
    yp, tp = out['y_hat'], out['state_pred']
    np.testing.assert_allclose(out['diagnostics_vec'], [yp.mean(), yp.std(), tp[:, -1].mean(), tp[:, -1].std()])
    err = np.abs(tp - T[jj])
    errors.append([np.mean(abs(yp-y[jj])), err.mean(), err[:, -1].mean(), *err.mean(0)])
np.set_printoptions(precision=5)
print('last16, contiguous 1-4, norm, axis 1-4; scalar, trajectory mean, endpoint:')
print(np.asarray(errors)[:, :3]); print('worst and average', np.max(errors, axis=0), np.mean(errors, axis=0))
