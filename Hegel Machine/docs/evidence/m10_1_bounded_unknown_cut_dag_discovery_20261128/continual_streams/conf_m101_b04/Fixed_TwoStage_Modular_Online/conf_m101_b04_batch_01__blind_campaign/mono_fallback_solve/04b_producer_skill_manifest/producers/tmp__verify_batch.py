import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; rng=np.random.default_rng(123)
folds=[np.arange(72)]+[rng.permutation(72) for _ in range(11)]
for k,f in enumerate(folds):
 i,j=f[:52],f[52:]; o=m.solve_batch(dict(s,train=dict(x=x[i],y=y[i],traj=t[i]),test=dict(x=x[j]))); e=abs(o['y_hat']-y[j]); et=abs(o['state_pred']-t[j]); print(k,round(e.mean(),5),round(et.mean(),5),round(et[:,-1].mean(),5),round(e.max(),5));
 if k==0: assert e.mean()<=.038
