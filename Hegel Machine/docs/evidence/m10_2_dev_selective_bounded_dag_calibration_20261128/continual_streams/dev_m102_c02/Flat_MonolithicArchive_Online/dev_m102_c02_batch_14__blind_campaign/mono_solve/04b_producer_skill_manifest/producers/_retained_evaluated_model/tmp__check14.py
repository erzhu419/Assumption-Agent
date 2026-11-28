import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']
for j,test in enumerate([np.arange(52,72), *np.array_split(np.random.default_rng(11).permutation(72),4), *[np.argsort(x[:,i])[-20:] for i in range(4)], *[np.argsort(x[:,i])[:20] for i in range(4)]]):
 train=np.setdiff1d(np.arange(len(x)),test); o=mod.solve_batch(dict(s,train={k:v[train] for k,v in tr.items() if k!='summary'},test={'x':x[test]})); ey=abs(y[test]-o['y_hat']); et=abs(t[test]-o['state_pred']); print(j,round(ey.mean(),6),round(et.mean(),6),round(et.mean(0).max(),6),round(et[:,-1].mean(),6))
