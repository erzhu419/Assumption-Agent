import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
a=s['train']; z=m._calibration(s['panel_a'],a['x']); n=len(z); times=len(s['eval_times'])
ids=[np.arange(52,72),np.argsort(abs(a['y']))[-20:],*[np.argsort(d)[k] for d in z.T for k in (slice(0,20),slice(-20,None))],*[np.random.default_rng(i).permutation(n)[:20] for i in range(16)]]
def verify(o,n):
 y,p,d=(o[k] for k in ('y_hat','state_pred','diagnostics_vec')); assert y.shape==(n,) and p.shape==(n,times) and d.shape==(4,); assert all(np.isfinite(v).all() for v in (y,p,d)); np.testing.assert_allclose(d,[y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()]);return y,p
metrics=[]
for va in ids:
 tr=np.setdiff1d(np.arange(n),va)
 y,p=verify(m.solve_batch(dict(s,train={k:v[tr] if k!='summary' else v for k,v in a.items()},test={'x':a['x'][va]})),len(va))
 e=np.abs(p-a['traj'][va]); metrics.append([np.mean(abs(y-a['y'][va])),e.mean(),e.mean(0).max(),e.max()])
print('52/20 [scalar MAE, state MAE, worst horizon MAE, max state point error]:',metrics[0]); print('Worst across',len(ids),'slices:',np.max(metrics,axis=0)); assert np.max(np.asarray(metrics)[:,:3])<.038
verify(m.solve_batch(s),len(s['test']['x'])); print('Public test shapes, finiteness and diagnostics verified in isolated process')
