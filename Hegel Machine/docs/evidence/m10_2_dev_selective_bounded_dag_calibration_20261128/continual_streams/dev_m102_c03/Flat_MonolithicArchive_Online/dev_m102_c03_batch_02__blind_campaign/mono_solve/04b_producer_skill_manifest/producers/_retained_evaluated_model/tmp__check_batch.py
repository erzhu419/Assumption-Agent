import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['x']); rng=np.random.default_rng(102)
orders=[np.arange(N),np.arange(N)[::-1]]+[rng.permutation(N) for _ in range(24)]
for j in range(4): orders += [np.argsort(tr['x'][:,j]),np.argsort(-tr['x'][:,j])]
errs=[]
for k,order in enumerate(orders):
 fit,val=order[:52],order[52:]; pred=m.solve_batch({**s,'train':{**tr,**{key:tr[key][fit] for key in ('x','y','traj')}},'test':{'x':tr['x'][val]}})
 ey=np.abs(pred['y_hat']-tr['y'][val]); et=np.abs(pred['state_pred']-tr['traj'][val]); errs.append([ey.mean(),et.mean(),et.mean(axis=0).max(),ey.max(),et.mean(axis=1).max()]);
 if k==0: print('52/20 scalar, trajectory, worst horizon, worst scalar row, worst trajectory row',errs[-1]); print('per horizon',et.mean(axis=0))
print('random/reversed maxima',np.max(errs[:26],axis=0)); print('coordinate tail holdouts',np.asarray(errs[26:]));
p=m.solve_batch(s); assert all(np.isfinite(a).all() for a in p.values()); print('public diagnostics',p['diagnostics_vec']);
