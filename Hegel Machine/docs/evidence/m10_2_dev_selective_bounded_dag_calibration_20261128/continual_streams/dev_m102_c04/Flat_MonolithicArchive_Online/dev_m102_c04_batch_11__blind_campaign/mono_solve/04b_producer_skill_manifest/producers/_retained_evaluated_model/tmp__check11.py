import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=map(tr.get,('x','y','traj'))
z=m._calibrate(s['panel_a'],x); p,r=m._operators(s['panel_c'],z); ij=[(i,j) for i in range(4) for j in range(4) if i!=j]; b=m._design(p,r,np.sum(p*r,axis=1)); c=np.column_stack([p[:,i]*r[:,j] for i,j in ij]); f=np.column_stack((b,c)); q=m._time_basis(s['eval_times'],T[:52]); a=T@q
print('cross coeff',list(zip(ij,np.round(m._fit(f[:52],a[:52])[-12:],4))))
print('scalar',m._fit(b[:52],y[:52]))
for label,te in [('required',np.arange(52,72)), *[(f'fold{i}',v) for i,v in enumerate(np.array_split(np.random.default_rng(11).permutation(72),4))], *[(f'tail{i}',np.argsort(abs(z[:,i]))[-20:]) for i in range(4)]]:
 ix=np.setdiff1d(np.arange(len(x)),te); train={k:tr[k][ix] for k in ('x','y','traj')}; train['summary']=np.array([y[ix].mean(),y[ix].std(),T[ix,-1].mean(),T[ix,-1].std()]); res=m.solve_batch(dict(s,train=train,test={'x':x[te]})); e=abs(res['state_pred']-T[te]); print(label,round(np.mean(abs(res['y_hat']-y[te])),6),round(np.mean(e),6),round(np.max(e.mean(0)),6))
