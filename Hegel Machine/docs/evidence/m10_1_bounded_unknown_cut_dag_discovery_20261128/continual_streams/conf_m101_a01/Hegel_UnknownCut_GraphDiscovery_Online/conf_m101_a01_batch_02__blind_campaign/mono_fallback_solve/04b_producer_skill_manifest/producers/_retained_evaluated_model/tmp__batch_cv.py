import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,z=(s['train'][k] for k in ('x','y','traj'))
folds=[(np.arange(n),np.arange(n,len(x))) for n in (52,56)]+[(np.setdiff1d(np.arange(len(x)),j),j) for j in np.array_split(np.random.default_rng(26).permutation(len(x)),4)]
if __name__=='__main__':
 for tr,te in folds:
  sub=dict(s,train=dict(x=x[tr],y=y[tr],traj=z[tr]),test=dict(x=x[te])); p=solve_batch(sub); yerr=abs(p['y_hat']-y[te]); terr=abs(p['state_pred']-z[te]); print(len(tr), 'y MAE', yerr.mean(), 'state MAE', terr.mean(), 'last', terr[:,-1].mean(), 'worst time', terr.mean(0).max()); assert np.all(np.isfinite(p['diagnostics_vec']))
