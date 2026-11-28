import pickle
import numpy as np
from runpy import run_path
root='/holdout_task/'
n0=run_path(root+'_frozen_nodes/node_00.py')['run_node']
n1=run_path(root+'node_01.py')['run_node']
n2=run_path(root+'node_02.py')['run_node']
st=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; tr=st['train']; x,y,t=(tr[k] for k in ('x','y','traj')); i=np.arange(len(x)); rng=np.random.default_rng(235)
folds={'interleaved':[i%6==k for k in range(6)],'contiguous':[i//12==k for k in range(6)],'random':[np.isin(i,rng.choice(i,20,replace=False)) for _ in range(30)],'feature_tails':[(x[:,j] <= np.quantile(x[:,j],.2) if s<0 else x[:,j]>=np.quantile(x[:,j],.8)) for j in range(x.shape[1]) for s in [-1,1]],'public':[i>=52]}
for name, masks in folds.items():
 rows=[]
 for v in masks:
  train={'x':x[~v],'y':y[~v],'traj':t[~v]}; study=dict(st,train=train,test={'x':x[v]}); p0=n0({'__input__':study}); p1=n1({'__input__':study,'node_00':p0}); out=n2({'node_00':p0,'node_01':p1}); rows.append([np.mean(abs(out['y_hat']-y[v])),np.mean(abs(out['state_pred']-t[v]))]); assert all(np.all(np.isfinite(z)) for z in out.values())
 rows=np.array(rows); print(name,'mean',rows.mean(axis=0).round(6),'max',rows.max(axis=0).round(6),'worst endpoint fold',np.argmax(rows[:,0]))
