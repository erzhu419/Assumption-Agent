import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(712)
def check(train,test, label):
 st={**s,'train':{**tr,**{k:tr[k][train] for k in ('x','y','traj')}},'test':{'x':tr['x'][test]}}
 pred=m.solve_batch(st); ey=abs(pred['y_hat']-tr['y'][test]); et=abs(pred['state_pred']-tr['traj'][test]); print(label, np.round([ey.mean(),et.mean(),ey.max(),et.mean(0).max(),et.max()],6)); return ey.mean(),et.mean(),et.mean(0).max()
i=np.arange(n); rows=[check(i[:52],i[52:],'prescribed')]
for k in range(4):
 j=rng.permutation(n); rows.append(check(j[:52],j[52:],f'random {k}'))
z=m._calibration(s['panel_a'])(tr['x']);
for k in range(4):
 j=np.argsort(abs(z[:,k])); rows.append(check(j[:52],j[52:],f'axis {k} tail'))
print('worst MAE y, trajectory, single timestamp',np.max(rows,axis=0))
assert np.max(rows) <= .038
out=m.solve_batch(s); y=out['y_hat']; state=out['state_pred']; assert y.shape==(36,) and state.shape==(36,8); assert all(np.isfinite(v).all() for v in out.values()); np.testing.assert_allclose(out['diagnostics_vec'], [y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()]); print('clean-process output verified')
