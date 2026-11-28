import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); idx=np.arange(n)
a=mod._affine(s['panel_a'],robust=True); b=mod._affine(s['panel_b']); z=np.linalg.solve(a[1:].T,(tr['x']-a[0]).T).T@b[1:]+b[0]
def evaluate(te):
 fit=np.setdiff1d(idx,te); subset=dict(s,train={**tr,**{k:tr[k][fit] for k in ('x','y','traj')}},test={'x':tr['x'][te]}); out=mod.solve_batch(subset)
 e=np.abs(out['y_hat']-tr['y'][te]); et=np.abs(out['state_pred']-tr['traj'][te]); q=out['state_pred'][:,-1]; y=out['y_hat']; assert np.allclose(out['diagnostics_vec'],[y.mean(),y.std(),q.mean(),q.std()]); assert np.isfinite(et).all() and np.isfinite(e).all()
 return np.r_[e.mean(),et.mean(),et[:,-1].mean(),et.mean(0).max(),e.max()]
for name,te in [('prescribed',idx[52:]),('first',idx[:20]),('middle',idx[20:40])]: print(name,evaluate(te))
rng=np.random.default_rng(729); vals=np.array([evaluate(rng.choice(n,20,replace=False)) for _ in range(32)]); print('random worst',vals.max(0))
for j in range(4):
 order=np.argsort(z[:,j]); print('edge',j,'low',evaluate(order[:10]),'high',evaluate(order[-10:]))
order=np.argsort(np.linalg.norm(z,axis=1)); print('largest radius',evaluate(order[-12:])); print('test/prior range',np.min(z,0),np.max(z,0)); print('test latent range',np.min(np.linalg.solve(a[1:].T,(s['test']['x']-a[0]).T).T@b[1:]+b[0],0),np.max(np.linalg.solve(a[1:].T,(s['test']['x']-a[0]).T).T@b[1:]+b[0],0)); assert evaluate(idx[52:])[0]<=.038
