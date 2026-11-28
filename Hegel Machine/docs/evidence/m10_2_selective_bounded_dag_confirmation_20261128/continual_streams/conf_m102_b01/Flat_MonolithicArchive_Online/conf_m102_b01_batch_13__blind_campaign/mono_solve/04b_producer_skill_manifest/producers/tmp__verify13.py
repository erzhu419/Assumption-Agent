import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; rng=np.random.default_rng(321)
for label, order in [('official',np.arange(72)),('reverse',np.arange(71,-1,-1)),*[(f'random{i}',rng.permutation(72)) for i in range(20)]]:
 a,b=order[:52],order[52:]; tr={k:t[k][a] for k in ('x','y','traj')}; y,z=tr['y'],tr['traj'][:,-1]; tr['summary']=np.array([y.mean(),y.std(),z.mean(),z.std()]); o=m.solve_batch(dict(s,train=tr,test={'x':t['x'][b]})); ey=abs(o['y_hat']-t['y'][b]); et=abs(o['state_pred']-t['traj'][b]); print(label,*map(lambda f:round(float(f),7),[ey.mean(),et.mean(),et.mean(0).max(),ey.max(),et.max()])); assert ey.mean()<=.038 and et.mean(0).max()<=.038
