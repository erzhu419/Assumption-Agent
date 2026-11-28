import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; r=np.random.default_rng(481); n=len(d['x']); allidx=np.arange(n)
orders=[allidx]+[r.permutation(n) for _ in range(12)]
z=m._calibrate(s['panel_a'],d['x'])
orders += [np.argsort(side*z[:,k]) for k in range(z.shape[1]) for side in (-1,1)]
for j,o in enumerate(orders):
 a,b=o[:52],o[52:]; q=dict(s,train={k:v[a] if k!='summary' else v for k,v in d.items()},test={'x':d['x'][b]}); p=m.solve_batch(q)
 ey=abs(p['y_hat']-d['y'][b]); es=abs(p['state_pred']-d['traj'][b]); print(j,'y / traj / worst time / final',np.round([ey.mean(),es.mean(),es.mean(0).max(),es[:,-1].mean()],6))
