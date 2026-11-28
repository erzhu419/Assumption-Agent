import pickle, runpy
import numpy as np
study = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
upstream = runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':study})
node = runpy.run_path('/holdout_task/node_01.py')['run_node']
tr = study['train']; all_ids = np.arange(len(tr['x']))
folds = [all_ids[-20:]] + list(np.array_split(np.random.default_rng(123).permutation(len(all_ids)), 5))
for x in tr['x'].T:
    ix = np.argsort(x)
    folds.extend([ix[:12], ix[-12:]])
errors = []
for test in folds:
    train = np.setdiff1d(all_ids, test)
    inp = {**study, 'train':{k:tr[k][train] for k in ('x','y','traj')}, 'test':{'x':tr['x'][test]}}
    out = node({'__input__':inp, 'node_00':upstream})
    errors.append([np.mean(abs(out['y_hat']-tr['y'][test])),np.mean(abs(out['state_pred']-tr['traj'][test]))])
errors = np.array(errors)
print('Final artifact split MAEs (scalar, trajectory):', np.round(errors,6).tolist())
print('Worst standard split:', np.round(errors[:6].max(axis=0),6).tolist(), 'Worst coordinate-tail split:',np.round(errors[6:].max(axis=0),6).tolist())
assert np.isfinite(errors).all() and np.all(errors[:6].max(axis=0) <= .038)
