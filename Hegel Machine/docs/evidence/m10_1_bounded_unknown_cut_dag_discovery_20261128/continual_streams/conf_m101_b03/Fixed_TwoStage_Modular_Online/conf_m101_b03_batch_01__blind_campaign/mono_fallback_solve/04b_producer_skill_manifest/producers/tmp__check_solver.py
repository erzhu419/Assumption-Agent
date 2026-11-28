import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; idx=np.arange(len(tr['y'])); rng=np.random.RandomState(111)
parts=np.array_split(rng.permutation(idx),4); splits=[idx[52:],idx[56:],*parts]
for b in splits:
 a=np.setdiff1d(idx,b); v=m.solve_batch(dict(s, train={**tr,**{k:tr[k][a] for k in ['x','y','traj']}},test={'x':tr['x'][b]})); ey=abs(v['y_hat']-tr['y'][b]); et=abs(v['state_pred']-tr['traj'][b]); print(f'n={len(b)} y={ey.mean():.6f} traj={et.mean():.6f} worst_time={et.mean(0).max():.6f}',flush=True)
