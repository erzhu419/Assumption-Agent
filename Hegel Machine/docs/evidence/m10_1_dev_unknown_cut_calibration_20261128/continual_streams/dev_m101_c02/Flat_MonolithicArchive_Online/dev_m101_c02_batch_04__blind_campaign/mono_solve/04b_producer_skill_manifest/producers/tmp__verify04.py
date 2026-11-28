import importlib.util, pickle
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t = s['train']
new = load('/holdout_task/batch_solver.py')
base = load('/tmp/baseline04.py')
w = new._calibrate(s['panel_a'], t['x'])
q = new._calibrate(s['panel_a'], s['test']['x'])
order = np.random.default_rng(194).permutation(len(w))
edge = np.argsort(abs(w).max(1))[-16:]
endpoint = np.argsort(abs(t['traj'][:, -1]))[-16:]
slices = [np.arange(56,72), *np.array_split(order,5), edge, endpoint]
for mod in (base, new):
    errs = []
    for j in slices:
        i = np.setdiff1d(order, j)
        y, state = mod._predict(s['panel_b'], w[i], w[j], t['y'][i], t['traj'][i])
        e = abs(state - t['traj'][j]); errs.append([abs(y-t['y'][j]).mean(), e.mean(), e[:, -1].mean(), e.mean(axis=0).max()])
    print('baseline' if mod is base else 'challenger', 'MAE [scalar, trajectory, endpoint, worst timestamp]')
    print('56/16:', errs[0], 'amplitude edge:', errs[-2], 'large endpoint:', errs[-1]); print('Worst:',np.max(errs,axis=0))
    if mod is new:
        r=mod.solve_batch(s); y=r['y_hat']; state=r['state_pred']; end=state[:,-1]
        assert y.shape == (len(q),) and state.shape == (len(q),len(s['eval_times']))
        assert all(np.isfinite(v).all() for v in r.values())
        np.testing.assert_array_equal(r['diagnostics_vec'], [y.mean(),y.std(),end.mean(),end.std()])
        print('max calibrated test and train amplitude:', abs(q).max(), abs(w).max())
