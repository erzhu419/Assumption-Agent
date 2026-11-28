import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); w,q=m._coords(s,tr['x']); c=s['panel_c']; r,co=m._c_fit(c,s['telemetry_vec']); print('C1 rates, RMSE',r,np.std(m._c_basis(c['u'],r)@co-c['v1']));print('C1 coefficients\n',np.round(co,5))
rng=np.random.default_rng(98); splits=[np.arange(n)]+[rng.permutation(n) for _ in range(16)]; val=[]
for ix in splits:
 a,b=ix[:52],ix[52:]; ss=dict(s,train={k:v[a] if k in ('x','y','traj') else v for k,v in tr.items()},test={'x':tr['x'][b]}); pred=m.solve_batch(ss); ey=abs(pred['y_hat']-tr['y'][b]); ep=abs(pred['state_pred']-tr['traj'][b]); val.append([ey.mean(),ep.mean(),ep.mean(0).max(),ey.max(),ep.max()]);
print('52/20 scalar, traj, worst time, peak scalar, peak traj:',val[0]); print('random split worsts:',np.max(val[1:],axis=0));
for target in ('y','traj'):
 for name,D in [('q',q),('q+w',np.c_[q,w]),('q+opp',np.c_[q,m._interactions(q)]),('q+w+opp',np.c_[q,w,m._interactions(q)])]:
  d=m._aff(D); v=tr[target]; a=np.linalg.lstsq(d[:52],v[:52],rcond=None)[0]; print(target,name,np.mean(abs(d[52:]@a-v[52:])),np.std(d[:52]@a-v[:52]),'coef',np.round(a,4) if target=='y' else np.round(np.linalg.norm(a,axis=1),4))
