import pickle, runpy
import numpy as np

root = '/holdout_task/'
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
cal = runpy.run_path(root + '_frozen_nodes/node_00.py')['run_node']({'__input__': s})
mix = runpy.run_path(root + '_frozen_nodes/node_01.py')['run_node']({'node_00': cal})
sink = runpy.run_path(root + 'node_02.py')['run_node']
x, z = cal['cal_tr'], mix['z_tr']
y, tr = s['train']['y'], s['train']['traj']
ids = np.arange(len(y))

def evaluate(name, slices):
    out = []
    for te in slices:
        fit = np.setdiff1d(ids, te)
        pred = sink({'__input__': {'train': {'y': y[fit], 'traj': tr[fit]}},
                     'node_00': {'cal_tr': x[fit], 'cal_te': x[te]},
                     'node_01': {'z_tr': z[fit], 'z_te': z[te]}})
        err = np.abs(pred['state_pred'] - tr[te])
        out.append([np.mean(np.abs(pred['y_hat'] - y[te])), err.mean(), err.mean(axis=0).max()])
    out = np.asarray(out)
    print(name, 'mean', out.mean(axis=0).round(5), 'worst', out.max(axis=0).round(5), 'worst-label slice', out[:,0].argmax())

evaluate('original shuffled', np.array_split(np.random.default_rng(7).permutation(ids), 4))
evaluate('fresh shuffled', [fold for seed in (41, 113, 307) for fold in np.array_split(np.random.default_rng(seed).permutation(ids), 4)])
evaluate('contiguous', np.array_split(ids, 4))
evaluate('coordinate tails', [np.argsort(v)[::direction][:12] for v in np.column_stack((x,z)).T for direction in (1,-1)])
