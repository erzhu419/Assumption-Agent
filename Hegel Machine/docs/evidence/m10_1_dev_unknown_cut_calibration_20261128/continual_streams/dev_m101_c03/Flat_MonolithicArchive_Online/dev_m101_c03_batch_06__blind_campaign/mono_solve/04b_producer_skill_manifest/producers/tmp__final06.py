import sys, pickle
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch, _decode
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr = s['train']
x, y, traj = (tr[k] for k in ('x','y','traj'))
n = len(x)
r = np.linalg.norm(_decode(s['panel_a'],x),axis=1)
rng = np.random.default_rng(42)
splits = {'56/16':np.arange(56,n), **{f'random{k}':rng.permutation(n)[:16] for k in range(4)}, 'outer':np.argsort(r)[-16:], 'inner':np.argsort(r)[:16]}
metrics = []
for key, j in splits.items():
    i = np.setdiff1d(np.arange(n),j)
    sub = dict(s, train=dict(tr, x=x[i], y=y[i], traj=traj[i]), test={'x':x[j]})
    p = solve_batch(sub)
    yh, st = p['y_hat'], p['state_pred']
    assert yh.shape == y[j].shape and st.shape == traj[j].shape
    assert all(np.isfinite(v).all() for v in p.values())
    np.testing.assert_allclose(p['diagnostics_vec'],[yh.mean(),yh.std(),st[:,-1].mean(),st[:,-1].std()])
    e = np.abs(st-traj[j])
    metrics.append([np.abs(yh-y[j]).mean(),e.mean(),e.mean(0).max(),e[:,-1].mean()])
    print(key, np.round(metrics[-1],6))
w = np.max(metrics,axis=0)
print('Worst MAE: endpoint, trajectory, horizon, terminal', w.round(6))
assert np.all(w < [.007,.007,.010,.010]), w
