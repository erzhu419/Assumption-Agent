import pickle,importlib.util,sys,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];rng=np.random.default_rng(18)
folds=[np.arange(72)]+[rng.permutation(72) for _ in range(9)]
for ids in folds:
 a,b=ids[:52],ids[52:]; q=dict(s,train={k:v[a] for k,v in t.items() if k!='summary'},test={'x':t['x'][b]});o=m.solve_batch(q); ey=abs(o['y_hat']-t['y'][b]); et=abs(o['state_pred']-t['traj'][b]);print(np.round([ey.mean(),et.mean(),et.mean(0).max(),np.quantile(ey,.9),np.quantile(et.mean(1),.9)],5))
