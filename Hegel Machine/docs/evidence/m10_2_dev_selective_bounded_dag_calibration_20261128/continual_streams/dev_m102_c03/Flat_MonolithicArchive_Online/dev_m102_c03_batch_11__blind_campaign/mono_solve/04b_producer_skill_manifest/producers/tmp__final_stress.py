import pickle, importlib.util, numpy as np
f='/holdout_task/batch_solver.py'; sp=importlib.util.spec_from_file_location('solver',f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as fp: s=pickle.load(fp)[0]['study']
t=s['train']; x=t['x']; rng=np.random.default_rng(12328)
def score(ids):
 a,b=ids[:52],ids[52:]; tr={k:t[k][a] for k in ('x','y','traj')}; tr['summary']=np.array([tr['y'].mean(),tr['y'].std(),tr['traj'][:,-1].mean(),tr['traj'][:,-1].std()]); o=m.solve_batch(dict(s,train=tr,test={'x':x[b]})); y,H=o['y_hat'],o['state_pred']; assert y.shape==(20,) and H.shape==(20,8) and np.isfinite(H).all() and np.isfinite(y).all(); np.testing.assert_allclose(o['diagnostics_vec'],[y.mean(),y.std(),H[:,-1].mean(),H[:,-1].std()]); E=abs(H-t['traj'][b]); return np.array([np.mean(abs(y-t['y'][b])),E.mean(),E.mean(0).max()])
pub=score(np.arange(72)); rnd=np.array([score(rng.permutation(72)) for i in range(40)]); radial=score(np.argsort(np.linalg.norm(x-x.mean(0),axis=1))); edges=np.array([score(np.argsort(sign*x[:,j])) for j in range(4) for sign in (1,-1)]); print('52/20 scalar, trajectory, worst time:',pub); print('40 new shuffled splits max:',rnd.max(0)); print('highest-radius holdout:',radial); print('coordinate-tail holdouts max:',edges.max(0)); assert max(pub[:2])<=.038
