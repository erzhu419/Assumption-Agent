import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];N=len(tr['x']);R=np.random.default_rng(713)
folds=[np.arange(52,N)]+[R.permutation(N)[:20] for _ in range(12)]
metrics=[]
for j in folds:
 i=np.setdiff1d(np.arange(N),j)
 out=m.solve_batch(dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]}));e=abs(out['state_pred']-tr['traj'][j]);metrics.append([abs(out['y_hat']-tr['y'][j]).mean(),e.mean(),e[:,-1].mean(),*e.mean(0)])
a=np.array(metrics); print('52/20',a[0]); print('12 shuffled mean',a[1:].mean(0));print('12 shuffled max',a[1:].max(0))
