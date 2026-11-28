import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=(tr[k] for k in ('x','y','traj')); tm=s['eval_times']; ids=np.arange(len(x)); cal=m._calibrate(s['panel_a'],x)
high=np.argsort(np.max(abs(cal),axis=1))[-16:]; rng=np.random.default_rng(148)
folds=[('52/20',ids[52:]),('56/16',ids[56:]),('high-amplitude',high)]+[(f'random-{k}',j) for k,j in enumerate(np.array_split(rng.permutation(len(x)),4))]
metrics=[]
for name,j in folds:
 i=np.setdiff1d(ids,j); sub=dict(s,train={'x':x[i],'y':y[i],'traj':t[i],'summary':np.array([y[i].mean(),y[i].std(),t[i,-1].mean(),t[i,-1].std()])},test={'x':x[j]}); pred=m.solve_batch(sub)
 ym=np.mean(abs(pred['y_hat']-y[j])); err=abs(pred['state_pred']-t[j]); met=[ym,err.mean(),err.mean(0).max(),err[:,-1].mean()]; metrics.append(met)
 def basis(r): return -np.expm1(-tm[:,None]*r)
 def residual(r):
  b=basis(r); return (b@m._ls(b,t[i].T)-t[i].T).ravel()
 r=least_squares(residual,[1.8,.4],bounds=([.7,.05],[4.,.65])).x; b=np.linalg.qr(basis(r))[0]; err2=abs(pred['state_pred']@b@b.T-t[j]); print(name,'scalar, trajectory, worst horizon, endpoint MAE',np.round(met,6),'projected trajectory/worst horizon',np.round([err2.mean(),err2.mean(0).max()],6))
print('WORST scalar, trajectory, horizon, endpoint MAE',np.round(np.max(metrics,axis=0),6)); assert np.max(metrics)<.038
q=m.solve_batch(s); assert all(isinstance(v,np.ndarray) for v in q.values()); assert np.allclose(q['diagnostics_vec'],[q['y_hat'].mean(),q['y_hat'].std(),q['state_pred'][:,-1].mean(),q['state_pred'][:,-1].std()],atol=1e-12,rtol=0)
