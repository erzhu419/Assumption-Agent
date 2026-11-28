import importlib.util
import pickle
import numpy as np
path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr = s['train']
x, y, traj = (tr[k] for k in ('x','y','traj'))
rows = np.arange(len(x))
splits = [('52/20',rows[52:]),('56/16',rows[56:])]
for i, held in enumerate(np.array_split(np.random.default_rng(19).permutation(rows), 4)):
    splits.append((f'oof-{i}',held))
z = m._calibrate(s['panel_a'], x)
splits.append(('edge-16', np.argsort(np.max(abs(z),axis=1))[-16:]))
for name, held in splits:
    fit = np.setdiff1d(rows, held)
    o = m.solve_batch(dict(s, train=dict(tr,x=x[fit],y=y[fit],traj=traj[fit]), test={'x':x[held]}))
    assert o['y_hat'].shape == (len(held),) and o['state_pred'].shape == (len(held),len(s['eval_times']))
    ye = abs(o['y_hat']-y[held]); se = abs(o['state_pred']-traj[held]); ep = o['state_pred'][:,-1]
    assert np.allclose(o['diagnostics_vec'], [o['y_hat'].mean(), o['y_hat'].std(), ep.mean(), ep.std()])
    print(f'{name:8} y={ye.mean():.6f} trajectory={se.mean():.6f} worst_time={se.mean(axis=0).max():.6f}')
    assert ye.mean() <= .038 and se.mean(axis=0).max() <= .038
sc, st = m._designs(s, x)
print('scalar coefficients', m._fit(sc[:52],y[:52]).round(5))
print('trajectory mixed coefficients', m._fit(st[:52],traj[:52])[-16:,[2,-1]].reshape(4,4,2).round(5))
