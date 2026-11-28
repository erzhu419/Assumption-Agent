import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; allx=tr['x']; n=len(allx)
def check(label,a,b):
    z=m.solve_batch(dict(s,train={**tr,**{k:tr[k][a] for k in ('x','y','traj')}},test={'x':allx[b]}))
    dy=abs(z['y_hat']-tr['y'][b]); dt=abs(z['state_pred']-tr['traj'][b]); last=z['state_pred'][:,-1]; y=z['y_hat']; np.testing.assert_array_equal(z['diagnostics_vec'],[y.mean(),y.std(),last.mean(),last.std()]); return [dy.mean(),dt.mean(),dt.mean(0).max(),dy.max(),dt.max()]
def runs(label,splits):
    e=np.array([check(label,a,b) for a,b in splits]); print(label,'first',e[0],'worst',e.max(0)); assert e[:,:3].max() <= .038
idx=np.arange(n); runs('official',[(idx[:52],idx[52:])]); runs('random',[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(n) for i in range(40)]]); order=[np.argsort(allx[:,j]) for j in range(4)]+[np.argsort(np.linalg.norm(allx,axis=1))]; runs('input-extremes',[(p[:52],p[52:]) for q in order for p in (q,q[::-1])]); out=m.solve_batch(s); print({k:v.shape for k,v in out.items()}); print('diagnostics',out['diagnostics_vec'])
