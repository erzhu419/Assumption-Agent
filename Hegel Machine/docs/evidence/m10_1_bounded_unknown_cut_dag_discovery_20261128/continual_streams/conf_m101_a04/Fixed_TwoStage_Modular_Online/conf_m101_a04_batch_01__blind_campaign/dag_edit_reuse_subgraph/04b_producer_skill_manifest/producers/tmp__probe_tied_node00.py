import pickle, runpy
import numpy as np
root = '/holdout_task/'
m = runpy.run_path(root + 'node_00.py')
sink = runpy.run_path(root + '_frozen_nodes/node_01.py')['run_node']
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
a, c = s['train'], s['panel_c']
x = a['x']; rows = np.arange(len(x)); u = c['u']
coef = m['_calibration'](s['panel_a'])
z = np.linalg.solve(coef[1:].T, (x - coef[0]).T).T
cols = [0, 1, 3, 4, 6]
def f(q): return m['_bilinear'](q)[:, cols]
def g(q):
    k = 1.1*q
    return m['_affine'](q + (np.logaddexp(k, -k) - np.log(2))/1.1)
def loo(fun, target):
    return np.max(np.mean(np.abs([fun(u[i:i+1])[0] @ m['_fit'](fun(u[rows_c != i]), target[rows_c != i]) - target[i] for i in rows_c]), axis=0))
rows_c = np.arange(len(u))
print('worst panel LOO channel MAE, baseline vs constrained:', loo(m['_bilinear'], c['v1']), loo(f, c['v1']), loo(m['_logcosh_readout'], c['v2']), loo(g, c['v2']))
p = f(z) @ m['_fit'](f(u), c['v1']); q = g(z) @ m['_fit'](g(u), c['v2'])
h = m['run_node']({'__input__': dict(s, test={'x': x[:0]})})['h']
opts = {'baseline': h, 'constrained_same_scale': np.c_[p, 2*q], 'constrained_equal_scale': np.c_[p, q], 'constrained_sum': p + q}
folds = {'official': [rows[-20:]], 'contiguous': np.array_split(rows, 4), 'mixed': np.array_split(np.random.default_rng(13).permutation(rows), 4), 'tails': [v for j in range(x.shape[1]) for v in (np.argsort(x[:,j])[:10],np.argsort(x[:,j])[-10:])]}
for name, h in opts.items():
    summary = {}
    for family, splits in folds.items():
        errors = []
        for va in splits:
            tr = np.setdiff1d(rows, va)
            out = sink({'node_00': {'h':h[np.r_[tr,va]], 'n_train':len(tr), 'y':a['y'][tr], 'traj':a['traj'][tr]}})
            dy = abs(out['y_hat'] - a['y'][va]); dt = abs(out['state_pred'] - a['traj'][va])
            errors.append([dy.mean(), dt.mean(), dt.mean(axis=0).max()])
        summary[family] = np.max(errors,axis=0).round(6).tolist()
    print(name, summary)
