import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
assert len(tr['x'])==72

def check(a,b):
    ss=dict(s,train={k:tr[k][a] for k in ('x','y','traj')},test={'x':tr['x'][b]})
    out=m.solve_batch(ss)
    ey=np.abs(out['y_hat']-tr['y'][b]); et=np.abs(out['state_pred']-tr['traj'][b])
    ht=m._design(s,tr['x'][b]); h=m._design(s,tr['x'][a]); raw=np.abs(ht@m._fit(h,tr['traj'][a])-tr['traj'][b]);
    return np.array([ey.mean(),et.mean(),et.mean(axis=0).max(),ey.max(),et.max(),raw.mean()])
idx=np.arange(72)
r=check(idx[:52],idx[52:]); print('52/20 [y MAE, state MAE, worst time MAE, max y error, max state error, raw state MAE]',r)
assert max(r[:3])<=.038
rng=np.random.default_rng(6271)
rs=[]
for _ in range(40):
    j=rng.permutation(idx); rs.append(check(j[:52],j[52:]))
print('40 shuffled splits worst',np.max(rs,axis=0))
h=m._design(s,tr['x']); ht=m._design(s,s['test']['x']); lev=np.sum((ht@np.linalg.pinv(h))**2,axis=1); print('test leverage',np.quantile(lev,[0,.5,.9,1]))
for v in [np.linalg.norm(tr['x'],axis=1),*[np.abs(tr['x'][:,i]) for i in range(4)],np.sum((h@np.linalg.pinv(h))**2,axis=1)]:
    j=np.argsort(v); print('boundary slice',check(j[:52],j[52:]))
out=m.solve_batch(s); y=out['y_hat']; z=out['state_pred']; d=out['diagnostics_vec']; assert y.shape==(36,) and z.shape==(36,8) and d.shape==(4,); assert all(np.isfinite(v).all() for v in out.values()); np.testing.assert_array_equal(d,[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]);print('isolated schema and diagnostics verified',d)
