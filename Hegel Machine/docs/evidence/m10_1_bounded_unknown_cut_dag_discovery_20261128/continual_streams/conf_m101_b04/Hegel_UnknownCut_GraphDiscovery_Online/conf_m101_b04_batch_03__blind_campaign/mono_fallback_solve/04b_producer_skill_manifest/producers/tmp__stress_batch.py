import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']
x, y, traj = (t[k] for k in ('x', 'y', 'traj'))
all_i = np.arange(len(x))
cases = [('public52', all_i[52:])]
cases += [(f'random{k}', np.random.default_rng(k).permutation(len(x))[-20:]) for k in range(16)]
cases += [(f'axis{k}{sign:+}', np.argsort(sign*x[:, k])[-16:]) for k in range(4) for sign in (1, -1)]
cases += [('radial', np.argsort(np.linalg.norm(x-x.mean(0), axis=1))[-16:])]
metrics = []
for name, j in cases:
    i = np.setdiff1d(all_i, j)
    fit = dict(s, train=dict(x=x[i], y=y[i], traj=traj[i], summary=np.array([y[i].mean(), y[i].std(), traj[i,-1].mean(), traj[i,-1].std()])), test=dict(x=x[j]))
    o = mod.solve_batch(fit)
    e, et = abs(o['y_hat']-y[j]), abs(o['state_pred']-traj[j])
    metrics.append((name, e.mean(), et.mean(), et.mean(0).max()))
for name, a, b, c in metrics:
    print(f'{name:10s} y={a:.6f} traj={b:.6f} worst_time={c:.6f}')
print('worst', [max(metrics,key=lambda r:r[k]) for k in (1,2,3)])
