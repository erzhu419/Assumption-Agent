import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; ids=np.arange(len(x)); rng=np.random.default_rng(483)
def run(j,k):
 p=m.solve_batch(dict(s,train={a:tr[a][j] for a in ('x','y','traj')},test={'x':x[k]})); return abs(p['y_hat']-tr['y'][k]),abs(p['state_pred']-tr['traj'][k])
for i,k in enumerate([ids[52:],np.argsort(np.linalg.norm(x,axis=1))[-20:]]+[rng.permutation(ids)[:20] for _ in range(8)]):
 e,f=run(np.setdiff1d(ids,k),k); print(i, 'MAE',e.mean(),f.mean(),'max temporal MAE',f.mean(0).max(),'max scalar error',e.max()); assert np.isfinite(e).all() and np.isfinite(f).all();
 if i==0: assert max(e.mean(),f.mean()) <= .038
p=m.solve_batch(s); assert np.allclose(p['diagnostics_vec'],[p['y_hat'].mean(),p['y_hat'].std(),p['state_pred'][:,-1].mean(),p['state_pred'][:,-1].std()])
