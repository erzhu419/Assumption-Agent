import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; train=s['train']; ids=np.arange(len(train['y']));
for j in [ids[52:],ids[56:],*np.array_split(np.random.default_rng(13).permutation(len(ids)),4)]:
 i=np.setdiff1d(ids,j); d=dict(s,train={k:v[i] if k!='summary' else v for k,v in train.items()},test={'x':train['x'][j]}); pred=m.solve_batch(d); et=np.abs(pred['state_pred']-train['traj'][j]); print(len(i), 'y',np.mean(abs(pred['y_hat']-train['y'][j])), 'trajectory',et.mean(), 'horizons',et.mean(0).round(5))
