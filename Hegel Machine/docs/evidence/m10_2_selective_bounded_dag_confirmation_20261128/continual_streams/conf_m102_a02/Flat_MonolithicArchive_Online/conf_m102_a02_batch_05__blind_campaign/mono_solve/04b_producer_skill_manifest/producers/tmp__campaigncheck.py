import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train'];rng=np.random.default_rng(886)
ps=[np.arange(72)]+[rng.permutation(72) for _ in range(25)]+[np.argsort(t['x'][:,j]*sgn) for j in range(4) for sgn in (-1,1)]
E=[]
for p in ps:
 a,b=p[:52],p[52:];r=m.solve_batch(dict(s,train={k:t[k][a] for k in ['x','y','traj']},test={'x':t['x'][b]}));E.append([abs(r['y_hat']-t['y'][b]).mean(),abs(r['state_pred']-t['traj'][b]).mean(),abs(r['state_pred']-t['traj'][b]).mean(axis=0).max()]);assert np.all(np.isfinite(r['diagnostics_vec']))
print('gate',E[0]);print('random mean',np.mean(E[1:26],axis=0),'worst',np.max(E[1:26],axis=0));print('extreme slices',np.round(E[26:],5),'worst',np.max(E[26:],axis=0));print('test ranges',[(k,v.min(),v.max()) for k,v in m.solve_batch(s).items()]);
