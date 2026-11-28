import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=m.calibrate(s['panel_a'],x); rng=np.random.default_rng(1234)
def score(i):
 j=np.setdiff1d(np.arange(len(x)),i); ss=dict(s,train={k:tr[k][j] for k in ('x','y','traj')},test={'x':x[i]}); p=m.solve_batch(ss)
 e=abs(p['y_hat']-tr['y'][i]); f=abs(p['state_pred']-tr['traj'][i]); assert np.isfinite(e).all() and np.isfinite(f).all()
 return np.r_[e.mean(),f.mean(),f.mean(0)]
print('prescribed',score(np.arange(52,72)))
print('random max',np.max([score(rng.choice(72,20,False)) for _ in range(24)],0))
for k in range(5):
 i=np.argsort(z[:,k] if k<4 else np.linalg.norm(z,axis=1)); print('extreme',k,score(i[:20]),score(i[-20:]))
