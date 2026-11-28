import pickle, runpy, numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; f=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']; m=runpy.run_path(p+'node_01.py'); g=m['run_node']; base=runpy.run_path('/tmp/node_01_baseline.py')['run_node']; tr=s['train']; ids=np.arange(len(tr['x'])); folds=[ids[-20:],ids[:20],ids[26:46],*[np.random.default_rng(j).permutation(len(ids))[:20] for j in (726,11,82,112,31,65,211)]]
errs=[]
for va in folds:
 ti=np.setdiff1d(ids,va); q=dict(s,train={**tr, **{k:tr[k][ti] for k in ("x","y","traj")}},test={'x':tr['x'][va]}); d={'__input__':q,'node_00':f({'__input__':q})}; a=g(d); b=base(d); assert np.array_equal(a['y_hat'],b['y_hat']); assert a['state_pred'].shape==tr['traj'][va].shape and np.isfinite(a['state_pred']).all() and a['diagnostics_vec'].shape==(4,); errs.append([np.mean(abs(a['y_hat']-tr['y'][va])),np.mean(abs(a['state_pred']-tr['traj'][va])),np.mean(abs(b['state_pred']-tr['traj'][va]))]); print('scalar, trajectory, baseline',np.round(errs[-1],6),flush=True)
print('worst',np.max(errs,axis=0),'mean',np.mean(errs,axis=0))
