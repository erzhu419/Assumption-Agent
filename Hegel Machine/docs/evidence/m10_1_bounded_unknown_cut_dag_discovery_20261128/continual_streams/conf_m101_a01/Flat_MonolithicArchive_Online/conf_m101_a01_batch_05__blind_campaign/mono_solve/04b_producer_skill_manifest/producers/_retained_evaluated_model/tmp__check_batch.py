import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; ids=np.arange(len(x))
folds=[('52/20',ids[52:]),('56/16',ids[56:])]+[(f'seed{i}',np.random.default_rng(i).permutation(len(ids))[:20]) for i in range(8,16)]
folds += [('outer shell',np.argsort(np.linalg.norm(x,axis=1))[-16:])]
for label,te in folds:
 tr=np.setdiff1d(ids,te); o=solve_batch(dict(s, train={**t, **{k:t[k][tr] for k in ['x','y','traj']}}, test={'x':x[te]})); y=o['y_hat']; a=o['state_pred']; err=abs(a-t['traj'][te]); scalar=np.mean(abs(y-t['y'][te])); print(label,*(round(v,6) for v in [scalar,err.mean(),err.mean(0).max(),err[:,-1].mean()])); np.testing.assert_allclose(o['diagnostics_vec'],[y.mean(),y.std(),a[:,-1].mean(),a[:,-1].std()]); assert label!='52/20' or scalar<=.038
