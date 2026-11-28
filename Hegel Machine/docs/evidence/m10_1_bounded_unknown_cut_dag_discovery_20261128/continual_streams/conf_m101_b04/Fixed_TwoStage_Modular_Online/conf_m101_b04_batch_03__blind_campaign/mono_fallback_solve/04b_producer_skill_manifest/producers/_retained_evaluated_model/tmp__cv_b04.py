import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; a=m._calibration(s['panel_a']); z=(x-a[0])@np.linalg.inv(a[1:]); print('calibration',np.round(a,6)); rng=np.random.default_rng(710)
for label,idx in [('52/20',np.arange(72)),('56/16',np.arange(72)),*[(f'random {j}',rng.permutation(72)) for j in range(8)],*[(f'edge {j}',np.argsort(abs(z[:,j]))) for j in range(4)]]:
 n=56 if label=='56/16' or label.startswith('random') else 52; i,v=idx[:n],idx[n:]; out=m.solve_batch(dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':x[v]})); ey=abs(out['y_hat']-tr['y'][v]); et=abs(out['state_pred']-tr['traj'][v]); print(label,np.round([ey.mean(),et.mean(),et[:,-1].mean(),ey.max(),et.mean(axis=1).max()],6))
