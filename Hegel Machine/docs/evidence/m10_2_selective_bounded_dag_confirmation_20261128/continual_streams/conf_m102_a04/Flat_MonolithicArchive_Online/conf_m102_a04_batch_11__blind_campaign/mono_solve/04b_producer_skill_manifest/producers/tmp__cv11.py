import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']
rng=np.random.default_rng(9238)
folds={'official':np.arange(52,72),'first':np.arange(20),'center':np.arange(26,46),'high raw norm':np.argsort(np.linalg.norm(x,axis=1))[-20:],'high abs x0':np.argsort(np.abs(x[:,0]))[-20:]}
folds.update({f'random-{i}':rng.choice(len(x),20,replace=False) for i in range(6)})
for label,ix in folds.items():
 jt=np.setdiff1d(np.arange(len(x)),ix); si=dict(s,train={k:tr[k][jt] if k!='summary' else tr[k] for k in tr},test={'x':x[ix]}); out=m.solve_batch(si); ey=np.abs(out['y_hat']-tr['y'][ix]); es=np.abs(out['state_pred']-tr['traj'][ix]); print(label, 'y/state MAE',np.round([ey.mean(),es.mean()],7),'worst time',round(es.mean(0).max(),7),'max y',round(ey.max(),7))
