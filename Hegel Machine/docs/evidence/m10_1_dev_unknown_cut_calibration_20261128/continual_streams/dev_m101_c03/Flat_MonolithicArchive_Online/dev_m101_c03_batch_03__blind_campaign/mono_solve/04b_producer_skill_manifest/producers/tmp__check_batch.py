import sys,pickle
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch, _decode
with open('/holdout_task/public_batch.pkl','rb') as f:
    s=pickle.load(f)[0]['study']
tr=s['train']; i=np.arange(len(tr['x'])); z=_decode(s['panel_a'],tr['x']); radius=np.linalg.norm(z,axis=1)
holdouts=[('56/16',i[56:]), ('outer-radius',np.argsort(radius)[-16:])]
holdouts += [(f'random-{seed}-{k}',h) for seed in (562,1297) for k,h in enumerate(np.array_split(np.random.default_rng(seed).permutation(i),5))]
errors=[]
for label,v in holdouts:
    u=np.setdiff1d(i,v)
    split=dict(s,train={**tr,**{k:tr[k][u] for k in ('x','y','traj')}},test={'x':tr['x'][v]})
    p=solve_batch(split)
    e=np.abs(p['state_pred']-tr['traj'][v]).mean(0)
    error=np.r_[np.abs(p['y_hat']-tr['y'][v]).mean(),e.mean(),e[-1],e.max()]
    errors.append(error)
    print(label,np.round(error,6))
print('worst',np.round(np.max(errors,axis=0),6),'average',np.round(np.mean(errors,axis=0),6))
p=solve_batch(s)
assert p['y_hat'].shape==(36,) and p['state_pred'].shape==(36,8)
assert all(np.isfinite(v).all() for v in p.values())
y=p['y_hat']; e=p['state_pred'][:,-1]
np.testing.assert_array_equal(p['diagnostics_vec'],[y.mean(),y.std(),e.mean(),e.std()])
