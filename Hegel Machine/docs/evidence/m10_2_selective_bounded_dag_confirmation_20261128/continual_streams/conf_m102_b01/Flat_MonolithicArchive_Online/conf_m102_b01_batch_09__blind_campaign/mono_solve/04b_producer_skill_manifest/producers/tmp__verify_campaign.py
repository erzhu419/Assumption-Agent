import importlib.util, numpy as np, pickle
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; train=s['train']; rng=np.random.default_rng(9); n=len(train['x']); worst=np.zeros(8); stats=[]
for fold,test in enumerate([np.arange(52,72), *np.array_split(rng.permutation(n),4), *[np.argsort(np.linalg.norm(train['x'],axis=1))[-20:]], *[rng.choice(n,20,False) for _ in range(20)]]):
 idx=np.setdiff1d(np.arange(n),test); p=m.solve_batch(dict(s,train=dict(train,**{k:train[k][idx] for k in ['x','y','traj']}),test={'x':train['x'][test]})); ey=np.abs(p['y_hat']-train['y'][test]); ep=np.abs(p['state_pred']-train['traj'][test]); worst=np.maximum(worst,ep.mean(0)); stats.append([ey.mean(),ep.mean(),ep[:,-1].mean()]);
 if fold<6: print('fold',fold,'MAE (scalar, state, terminal)',stats[-1],'max scalar',ey.max())
 assert max(ey.mean(),ep.mean()) <= .038
p=m.solve_batch(s); y, st = p['y_hat'], p['state_pred']; assert y.shape==(36,) and st.shape==(36,8); np.testing.assert_allclose(p['diagnostics_vec'],[y.mean(),y.std(),st[:,-1].mean(),st[:,-1].std()],rtol=0,atol=1e-15); assert np.all(np.isfinite(np.c_[y,st])); print('worst fold',np.max(stats,axis=0),'worst per horizon',worst,'diagnostics',p['diagnostics_vec'])
