import sys,pickle,numpy as np
from sklearn.model_selection import KFold
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=(s['train'][k] for k in ('x','y','traj'))
folds=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
r=np.argsort(np.linalg.norm(x,axis=1)); folds.append((r[:-12],r[-12:]))
for i,j in folds:
 d=solve_batch(dict(s,train=dict(x=x[i],y=y[i],traj=z[i]),test=dict(x=x[j])))
 ye=abs(y[j]-d['y_hat']); ze=abs(z[j]-d['state_pred']); print(np.round([ye.mean(),ze.mean(),ye.max(),*ze.mean(0)],6))
