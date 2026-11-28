import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];
for seed in (None,0,1,2):
 p=np.arange(72) if seed is None else np.random.default_rng(seed).permutation(72);i,j=p[:52],p[52:];v=dict(s,train={k:(a[i] if k!='summary' else a) for k,a in tr.items()},test={'x':tr['x'][j]});o=m.solve_batch(v);e=abs(o['state_pred']-tr['traj'][j]);print(seed, np.mean(abs(o['y_hat']-tr['y'][j])),e.mean(),e.mean(0))
