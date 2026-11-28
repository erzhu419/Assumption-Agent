import importlib.util, pickle, sys
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x, y, z = (t[k] for k in ('x','y','traj'))
ix = np.arange(len(x))
splits = [('gate', ix[56:]), *[(f'fold{i+1}', a) for i,a in enumerate(np.array_split(np.random.default_rng(18).permutation(ix),4))], ('high-radius', np.argsort(np.linalg.norm(x, axis=1))[-16:])]
if '--extra' in sys.argv:
    splits += [(f'edge{j}{tag}', np.argsort(sign*x[:,j])[-16:]) for j in range(4) for tag,sign in (('+',1),('-',-1))]
e = []
for name, hold in splits:
    it = np.setdiff1d(ix, hold)
    tr = dict(x=x[it], y=y[it], traj=z[it], summary=np.array([y[it].mean(), y[it].std(), z[it,-1].mean(), z[it,-1].std()]))
    out = mod.solve_batch(dict(s, train=tr, test={'x':x[hold]}))
    ey, ez = abs(out['y_hat']-y[hold]), abs(out['state_pred']-z[hold]); m=[ey.mean(), ez.mean(), ez[:,-1].mean()]; e.append(m)
    print(name, np.round(m,6), 'time MAE', np.round(ez.mean(0),5))
print('mean', np.mean(e,0), 'worst', np.max(e,0))
