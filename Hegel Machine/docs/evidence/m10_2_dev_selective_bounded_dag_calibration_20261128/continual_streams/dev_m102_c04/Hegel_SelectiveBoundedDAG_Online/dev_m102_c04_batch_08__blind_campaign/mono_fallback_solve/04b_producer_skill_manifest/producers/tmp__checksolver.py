import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(79325)
for idx in [np.arange(72),*(rng.permutation(72) for _ in range(3))]:
 a,b=idx[:52],idx[52:]; st=dict(s,train={**tr,**{k:tr[k][a] for k in ('x','y','traj')}},test={'x':tr['x'][b]}); out=solve_batch(st); e=abs(out['state_pred']-tr['traj'][b]); print('CV',np.mean(abs(out['y_hat']-tr['y'][b])),e.mean(),e.mean(0).max())
