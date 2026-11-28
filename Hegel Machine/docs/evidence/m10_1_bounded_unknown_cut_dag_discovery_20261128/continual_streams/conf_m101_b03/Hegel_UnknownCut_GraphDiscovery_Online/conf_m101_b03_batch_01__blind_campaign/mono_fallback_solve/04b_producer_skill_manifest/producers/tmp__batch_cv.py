import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']; t = s['train']; x = t['x']
z = m._latent(s['panel_a'], x)
rng = np.random.default_rng(173)
splits = [('ordered52', np.arange(72), 52), ('ordered56', np.arange(72), 56)]
splits += [(f'random-{i}', rng.permutation(72), 52) for i in range(6)]
splits += [('radial-tail', np.argsort(np.linalg.norm(z, axis=1)), 56)]
splits += [(f'coordinate-{j}-{sign}', np.argsort(sign*z[:,j]), 56) for j in range(4) for sign in (-1,1)]
worst = np.zeros(3)
for name, ix, n in splits:
    a,b = ix[:n],ix[n:]; sub = dict(s, train={k:t[k][a] for k in ('x','y','traj')}, test={'x':x[b]})
    o = m.solve_batch(sub)
    Y = abs(o['y_hat']-t['y'][b]); R = abs(o['state_pred']-t['traj'][b]); e = [Y.mean(), R.mean(), R[:,-1].mean()]
    worst = np.maximum(worst,e); print(name, np.round(e,6), 'max horizon', round(R.mean(0).max(),6))
print('worst',worst)
