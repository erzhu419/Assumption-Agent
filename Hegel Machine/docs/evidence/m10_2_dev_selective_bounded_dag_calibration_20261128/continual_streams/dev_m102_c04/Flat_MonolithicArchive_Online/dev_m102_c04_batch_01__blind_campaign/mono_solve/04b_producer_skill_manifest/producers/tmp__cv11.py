import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task');from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=s['train']; print('ranges',d['x'].min(0),d['x'].max(0),s['test']['x'].min(0),s['test']['x'].max(0));
for tr,te in [(np.arange(52),np.arange(52,72)), (np.arange(20,72),np.arange(20)), (np.random.default_rng(21).permutation(72)[:52],np.random.default_rng(21).permutation(72)[52:])]:
 q={**s,'train':{**d,**{k:d[k][tr] for k in ('x','y','traj')}},'test':{'x':d['x'][te]}};p=solve_batch(q);e=abs(d['y'][te]-p['y_hat']); a=abs(d['traj'][te]-p['state_pred']); print('CV',e.mean(),a.mean(), 'errors', np.round(np.c_[te,e,a.mean(1)],3));
