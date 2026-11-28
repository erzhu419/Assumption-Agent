import importlib.util, pickle, numpy as np
path='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; all_ids=np.arange(len(x));
def score(it,ix):
 st={**s,'train':{**tr,**{k:tr[k][it] for k in ('x','y','traj')}},'test':{'x':x[ix]}}
 p=m.solve_batch(st); e=abs(p['y_hat']-tr['y'][ix]); h=abs(p['state_pred']-tr['traj'][ix]); assert np.isfinite(h).all() and np.isfinite(e).all(); return np.r_[e.mean(), h.mean(), h.mean(axis=0)]
folds={'gate': [all_ids[52:]],'random':[np.random.default_rng(i).permutation(72)[:20] for i in range(32)], 'coordinate tails':[j for k in range(4) for j in (np.argsort(x[:,k])[:20],np.argsort(x[:,k])[-20:])], 'radial tails':[np.argsort(np.linalg.norm(x,axis=1))[-20:]]}
for name, rows in folds.items():
 r=np.array([score(np.setdiff1d(all_ids,ix),ix) for ix in rows]); print(name,'mean',r.mean(0)[:2],'max',r.max(0)[:2], 'max time slice',r[:,2:].max()); assert r.max() < .038
fy,ft=m._features(s,x); coeff=m._ls(ft,tr['traj']); print('trajectory couplings by time',coeff[-2:]); print('state training RMSE',np.mean((ft@coeff-tr['traj'])**2)**.5)
