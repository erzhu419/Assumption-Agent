import importlib.util
import pickle
import numpy as np
from solver_challenger import scalar_predict, state_predict

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root+'public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x, y, j = (t[k] for k in ('x','y','traj'))
n = len(x); all_rows = np.arange(n)
radius = np.linalg.norm(x, axis=1)
folds = [('public', np.arange(52,n))]
folds += [(f'heldout_{i}', v) for i,v in enumerate(np.array_split(np.random.default_rng(37).permutation(n),4))]
folds += [('radius', np.argsort(radius)[-15:])]
folds += [(f'axis_{i}_{sign:+d}', np.argsort(sign*x[:, i])[-12:]) for i in range(x.shape[1]) for sign in (-1,1)]
rows = []
for name,v in folds:
    it = np.arange(52) if name == 'public' else np.setdiff1d(all_rows, v)
    st = dict(s, train=dict(t, x=x[it], y=y[it], traj=j[it]), test={'x':x[v]})
    o = m.solve_batch(st)
    yp = scalar_predict(x[it], x[v], y[it], m._poly)
    sp = state_predict(st, m._poly, m._ridge, m._aff)
    old = o['state_pred']
    _, _, vt = np.linalg.svd(j[it], full_matrices=False)
    blend = (old+sp)/2
    sm = [np.mean(abs(z-j[v]), axis=0) for z in (old, sp, blend, (blend@vt[:2].T)@vt[:2])]
    row = [np.mean(abs(z-y[v])) for z in (o['y_hat'], yp, .2*o['y_hat']+.8*yp)] + [b for z in sm for b in (z.mean(), z[-1], z.max())]
    rows.append(row)
    print(name, np.round(row,5))
print('columns: scalar baseline/Matern/blend; state baseline/rational/blend/projected blend, each avg/terminal/worst horizon')
print('worst original slices', np.round(np.max(rows[:6], axis=0),5))
print('worst directional slices', np.round(np.max(rows[6:], axis=0),5))
