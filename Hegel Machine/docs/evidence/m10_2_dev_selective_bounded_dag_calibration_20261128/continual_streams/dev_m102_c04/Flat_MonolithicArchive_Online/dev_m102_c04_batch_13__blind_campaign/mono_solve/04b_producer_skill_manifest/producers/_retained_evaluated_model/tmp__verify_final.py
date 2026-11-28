import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
t = s['train']; idx = np.arange(len(t['x'])); rng = np.random.default_rng(90827)
z = m['_calibrate'](s['panel_a'], t['x'])
splits = [('public', idx[52:])] + [(f'corner{j}', np.argsort(v)[-20:]) for j,v in enumerate([*z.T,*(-z).T,np.linalg.norm(z,axis=1)])] + [(f'random{j}',rng.permutation(idx)[52:]) for j in range(20)]
def check(study):
    out = m['solve_batch'](study)
    y, h, d = (out[k] for k in ('y_hat','state_pred','diagnostics_vec'))
    n = len(study['test']['x']); nt = len(study['eval_times'])
    assert all(isinstance(a,np.ndarray) and np.isfinite(a).all() for a in (y,h,d))
    assert (y.shape,h.shape,d.shape) == ((n,),(n,nt),(4,))
    assert np.allclose(d,[y.mean(),y.std(),h[:,-1].mean(),h[:,-1].std()])
    return y,h
check(s); rows = []
for name, te in splits:
    tr = np.setdiff1d(idx, te)
    y,h = check(dict(s, train=dict(x=t['x'][tr], y=t['y'][tr], traj=t['traj'][tr], summary=t['summary']), test={'x':t['x'][te]}))
    e = np.abs(h-t['traj'][te]); rows.append([np.mean(np.abs(y-t['y'][te])),e.mean(),e.mean(axis=0).max()])
rows = np.array(rows)
print('52/20 public MAEs [scalar, trajectory, worst timestamp]:', rows[0])
print('30-split worst MAEs:', rows.max(axis=0), [splits[j][0] for j in rows.argmax(axis=0)])
assert rows.max() <= .038
