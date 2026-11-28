import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; n=len(x)
rng=np.random.default_rng(2718)
ix=np.arange(n); extent=np.linalg.norm(m._coordinates(s['panel_a'],x),axis=1)
splits=[('52/20',ix[:52],ix[52:]),('56/16',ix[:56],ix[56:])]
for i in range(5):
 j=rng.permutation(n);splits.append((f'random {i}',j[:52],j[52:]))
j=np.argsort(extent);splits.append(('largest radii',j[:56],j[56:]))
for name,a,b in splits:
 ss=dict(s,train=dict(x=x[a],y=y[a],traj=t[a]),test=dict(x=x[b])); v=m.solve_batch(ss); ye=abs(v['y_hat']-y[b]);te=abs(v['state_pred']-t[b]); print(name,'MAE',np.round([ye.mean(),te.mean(),te[:,-1].mean(),te.mean(axis=0).max()],6))
