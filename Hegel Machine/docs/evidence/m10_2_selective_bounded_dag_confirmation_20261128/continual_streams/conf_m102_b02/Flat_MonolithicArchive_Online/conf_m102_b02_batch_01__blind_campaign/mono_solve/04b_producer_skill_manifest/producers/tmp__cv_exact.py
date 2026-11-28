import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'; sp=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['y']); rng=np.random.default_rng(218)
def score(i,j):
    d=dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]}); r=m.solve_batch(d); e=abs(r['state_pred']-tr['traj'][j]); return np.r_[abs(r['y_hat']-tr['y'][j]).mean(),e.mean(),e.mean(0)]
P=[np.arange(N)]+[rng.permutation(N) for _ in range(18)]; E=np.array([score(p[:52],p[52:]) for p in P]); print('public:',E[0]); print('worst column-wise:',E.max(0)); print('mean:',E.mean(0)); X=tr['x']; E=np.array([score(p[:52],p[52:]) for p in [np.argsort(sign*X[:,i]) for i in range(4) for sign in [-1,1]]]); print('directional extrapolation worst:',E.max(0)); print(E[:,:2]);
