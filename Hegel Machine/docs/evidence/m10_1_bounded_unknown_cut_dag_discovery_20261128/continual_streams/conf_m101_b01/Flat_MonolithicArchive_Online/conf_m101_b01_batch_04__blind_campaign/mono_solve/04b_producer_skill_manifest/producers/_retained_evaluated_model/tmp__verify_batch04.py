import importlib.util
import pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', p+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(p+'public_batch.pkl', 'rb'))[0]['study']
t = s['train']
x = t['x']
ids = np.arange(len(x))
rng = np.random.default_rng(420)
slices = [('52/20', ids[52:]), ('first20', ids[:20]), ('middle20', ids[24:44]), ('56/16', ids[56:])]
slices += [(f'random{k}', rng.choice(ids, 20, replace=False)) for k in range(6)]
q = m._coordinates(s, x)
d = ((q-q.mean(0))/q.std(0))
slices += [('outer12', np.argsort(np.linalg.norm(d,axis=1))[-12:])]
for name, j in slices:
    i = np.setdiff1d(ids, j)
    study = {**s, 'train': {**t, **{k:t[k][i] for k in ('x','y','traj')}}, 'test': {'x':x[j]}}
    out = m.solve_batch(study)
    y, state = out['y_hat'], out['state_pred']
    a = abs(y-t['y'][j]).mean(); e = abs(state-t['traj'][j]).mean(); term = abs(state[:,-1]-t['traj'][j,-1]).mean()
    print(f'{name:10s} y={a:.6f} state={e:.6f} terminal={term:.6f}')
    if name=='52/20': assert a <= .038
out = m.solve_batch(s)
y, state = out['y_hat'], out['state_pred']
assert np.allclose(out['diagnostics_vec'], [y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])
