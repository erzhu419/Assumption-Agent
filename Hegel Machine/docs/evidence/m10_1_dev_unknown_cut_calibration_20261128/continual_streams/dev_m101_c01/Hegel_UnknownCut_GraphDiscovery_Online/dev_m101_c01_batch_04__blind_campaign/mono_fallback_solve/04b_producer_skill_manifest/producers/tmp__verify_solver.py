import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; ids=np.arange(len(d['x'])); tail=np.argsort(np.sum(d['x']**2,axis=1))[-14:]; tests=[ids[56:],*np.array_split(np.random.default_rng(218).permutation(len(ids)),5),tail]; err=[]
for test in tests:
 tr=np.setdiff1d(ids,test); y=d['y'][tr]; z=d['traj'][tr]; end=z[:,-1]; q=m.solve_batch(dict(s,train=dict(x=d['x'][tr],y=y,traj=z,summary=np.array([y.mean(),y.std(),end.mean(),end.std()])),test={'x':d['x'][test]})); e=np.mean(abs(q['y_hat']-d['y'][test])); delta=abs(q['state_pred']-d['traj'][test]);err.append([e,delta.mean(),delta[:,-1].mean()]);
print('MAE y, trajectory, terminal:', np.array(err)); print('worst',np.max(err,axis=0)); print('worst random fold',np.max(err[1:-1],axis=0))
