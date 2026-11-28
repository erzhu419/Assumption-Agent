import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
tr=s['train']; result=mod.solve_batch(s); y,st=result['y_hat'],result['state_pred']; np.testing.assert_array_equal(result['diagnostics_vec'], [y.mean(), y.std(), st[:,-1].mean(), st[:,-1].std()]); print('test',y.shape,st.shape,result['diagnostics_vec']); assert np.isfinite(y).all() and np.isfinite(st).all()
z=mod._calibrate(s['panel_a'],tr['x']); ztest=mod._calibrate(s['panel_a'],s['test']['x']);print('test calibrated range',ztest.min(),ztest.max());
splits=[('required',np.arange(52,72))]+[(f'random{i}',np.random.default_rng(i).permutation(72)[52:]) for i in range(12)]+[(f'extreme{j}{side}',np.argsort(side*z[:,j])[-20:]) for j in range(4) for side in (-1,1)]; worst=np.zeros(3)
for name, val in splits:
    train=np.setdiff1d(np.arange(72),val); ss=dict(s,train={k:(v[train] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][val]}); p=mod.solve_batch(ss); e=np.abs(p['state_pred']-tr['traj'][val]); m=np.array([np.mean(np.abs(p['y_hat']-tr['y'][val])),e.mean(),e.mean(0).max()]);worst=np.maximum(worst,m); print(name, np.round(m,6)); assert max(m)<.038
print('worst y / trajectory / single horizon', worst)
