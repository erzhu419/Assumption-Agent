import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];rg=np.random.default_rng(550);splits=[np.arange(len(t['x']))]+[rg.permutation(len(t['x'])) for _ in range(8)]
for p in splits:
 a,b=p[:52],p[52:];out=solve_batch(dict(s,train={k:v[a] if k in ('x','y','traj') else v for k,v in t.items()},test={'x':t['x'][b]}));ey=np.abs(out['y_hat']-t['y'][b]);et=np.abs(out['state_pred']-t['traj'][b]);print('split',ey.mean(),et.mean(),np.round(et.mean(0),4), 'y max',ey.max())
