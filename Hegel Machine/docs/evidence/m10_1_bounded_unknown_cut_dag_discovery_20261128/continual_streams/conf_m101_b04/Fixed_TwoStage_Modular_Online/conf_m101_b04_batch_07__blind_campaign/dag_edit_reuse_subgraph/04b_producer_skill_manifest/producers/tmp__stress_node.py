import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
import sys
path=sys.argv[1] if len(sys.argv)>1 else p+'node_01.py'
node=runpy.run_path(path)['run_node']
n=len(z)
folds=list(np.array_split(np.random.default_rng(17).permutation(n),5))+[np.arange(n-20,n), np.argsort(np.max(abs(z),axis=1))[-15:]]
for i,h in enumerate(folds):
 j=np.setdiff1d(np.arange(n),h)
 d=node({'node_00':dict(a,z_tr=z[j],z_te=z[h],y=y[j],traj=T[j])})
 e=np.abs(d['state_pred']-T[h]); f=d['y_hat']; end=d['state_pred'][:,-1]
 np.testing.assert_allclose(d['diagnostics_vec'],[f.mean(),f.std(),end.mean(),end.std()]); assert np.isfinite(e).all() and np.isfinite(f).all()
 print(f'slice {i}: scalar {np.mean(abs(f-y[h])):.6f}, trajectory {e.mean():.6f}, worst horizon {e.mean(axis=0).max():.6f}')
