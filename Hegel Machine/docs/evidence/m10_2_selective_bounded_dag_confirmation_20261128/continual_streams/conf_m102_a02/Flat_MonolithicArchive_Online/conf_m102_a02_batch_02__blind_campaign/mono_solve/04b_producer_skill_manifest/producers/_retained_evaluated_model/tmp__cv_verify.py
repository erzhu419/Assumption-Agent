import numpy as np, pickle, importlib.util
sp=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py'); bs=importlib.util.module_from_spec(sp); sp.loader.exec_module(bs)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['y']); ids=np.arange(N)
sh,sc=bs._calibrate(s['panel_a']); z=(tr['x']-sh)/sc
rng=np.random.default_rng(45)
splits=[('gate',ids[:52],ids[52:])]+[(f'rand{k}',p[:52],p[52:]) for k in range(8) for p in [rng.permutation(ids)]]
for u in [abs(z[:,2]-.42*z[:,0]),np.linalg.norm(z,axis=1),z[:,2]]:
 order=np.argsort(u); splits.append(('extreme',order[:-12],order[-12:]))
for name,i,j in splits:
 ss=dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]}); o=bs.solve_batch(ss)
 e=abs(o['y_hat']-tr['y'][j]); r=abs(o['state_pred']-tr['traj'][j]); print(name, 'y MAE/max',round(e.mean(),6),round(e.max(),6),'state MAE/worst time',round(r.mean(),6),round(r.mean(axis=0).max(),6))
