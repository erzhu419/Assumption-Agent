import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
def check(a,b):
 sub=dict(s,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[b]})
 out=mod.solve_batch(sub); assert all(np.all(np.isfinite(v)) for v in out.values()); yy=out['y_hat']; st=out['state_pred']; assert yy.shape==(len(b),) and st.shape==(len(b),len(s['eval_times'])); np.testing.assert_allclose(out['diagnostics_vec'],[yy.mean(),yy.std(),st[:,-1].mean(),st[:,-1].std()]); e=np.abs(yy-tr['y'][b]); d=np.abs(st-tr['traj'][b]); return [e.mean(),d.mean(),d.mean(0).max(),e.max(),d.mean(1).max()]
print('fixed split: y MAE, trajectory MAE, worst horizon MAE, max y error, max row trajectory MAE',check(np.arange(52),np.arange(52,n)))
f=mod._design(s,x); leverage=np.sum(np.linalg.qr(f)[0]**2,axis=1); r=np.linalg.norm(x,axis=1); rng=np.random.default_rng(447)
rs=[]
for _ in range(100):
 ids=rng.permutation(n); rs.append(check(ids[:52],ids[52:]))
print('worst across 100 random splits',np.max(rs,axis=0))
for name,score in [('leverage',leverage),('input norm',r)]+[(f'x{j} {sgn}',sgn*x[:,j]) for j in range(4) for sgn in (-1,1)]:
 ids=np.argsort(score); print(name,'outermost 20',check(ids[:52],ids[52:]))
y=tr['y']; q=mod._basis(s['eval_times'],tr['traj']); B=mod._fit(f,np.c_[y,mod._fit(q,tr['traj'].T).T]); print('physical readout coefficients\n',np.round(B,3))
