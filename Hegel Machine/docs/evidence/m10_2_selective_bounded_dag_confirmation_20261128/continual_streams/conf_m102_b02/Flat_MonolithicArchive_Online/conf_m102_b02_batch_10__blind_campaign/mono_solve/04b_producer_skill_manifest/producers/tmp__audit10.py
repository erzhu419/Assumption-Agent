import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['x']); rng=np.random.default_rng(10)
for k in range(7):
 ids=np.arange(N) if k==0 else rng.permutation(N); a,b=ids[:52],ids[52:]; cv=dict(s,train={key:val if key=='summary' else val[a] for key,val in tr.items()},test={'x':tr['x'][b]}); o=m.solve_batch(cv); ry=np.abs(o['y_hat']-tr['y'][b]); rt=np.abs(o['state_pred']-tr['traj'][b]); print(k,'MAE',ry.mean(),rt.mean(),'worst time',rt.mean(0).max(),'worst row',ry.max(),rt.mean(1).max())
z,w,h,c=m._features(s,tr['x']); F=m._intercept(np.c_[z,h,c]); coef=m._ls(F,np.c_[tr['y'],tr['traj'][:,[1,-1]]]); print('coefficients rows = 1,z,h,z*h:');print(coef.round(4)); print('train residual std',np.std(F@coef-np.c_[tr['y'],tr['traj'][:,[1,-1]]],axis=0)); print('times',s['eval_times']); print('telemetry',s['telemetry_vec']);print('test coordinate ranges',[(round(a.min(),3),round(a.max(),3)) for a in m._features(s,s['test']['x'])]);
