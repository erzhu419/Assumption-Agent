import sys, importlib.util, pickle, numpy as np
path=sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); allidx=np.arange(n)
folds=[('required-tail',allidx[52:])]+[(f'block-{i}',j) for i,j in enumerate(np.array_split(allidx,4))]+[(f'axis-{i}-high',np.argsort(tr['x'][:,i])[-15:]) for i in range(4)]
for name,va in folds:
 ix=np.setdiff1d(allidx,va); r=dict(s,train={k:v[ix] if k in ('x','y','traj') else v for k,v in tr.items()},test={'x':tr['x'][va]}); p=m.solve_batch(r); y,st=p['y_hat'],p['state_pred']; diag=np.array([y.mean(),y.std(),st[:,-1].mean(),st[:,-1].std()]); assert np.allclose(diag,p['diagnostics_vec']) and np.isfinite(st).all() and np.isfinite(y).all(); print(name,np.round([np.mean(abs(y-tr['y'][va])),np.mean(abs(st-tr['traj'][va])), np.mean(abs(st[:,-1]-tr['traj'][va,-1]))],6))
