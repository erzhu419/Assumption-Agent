import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(991); z=m.calibrate(s['panel_a'],tr['x']); radius=np.max(abs(z),axis=1)
for name,ix in [('official',np.arange(52,72)),*[(f'block{i}',np.arange(i,i+20)) for i in (0,13,26,39)],*[(f'random{i}',rng.choice(72,20,False)) for i in range(6)],('extreme',np.argsort(radius)[-20:])]:
 it=np.setdiff1d(np.arange(72),ix); q={**s,'train':{k:v[it] if len(v)==72 else v for k,v in tr.items()},'test':{'x':tr['x'][ix]}}
 o=m.solve_batch(q); dy=abs(o['y_hat']-tr['y'][ix]); ds=abs(o['state_pred']-tr['traj'][ix]); print(name,dy.mean(),ds.mean(),ds.mean(0).max(),dy.max(),ds.max())
