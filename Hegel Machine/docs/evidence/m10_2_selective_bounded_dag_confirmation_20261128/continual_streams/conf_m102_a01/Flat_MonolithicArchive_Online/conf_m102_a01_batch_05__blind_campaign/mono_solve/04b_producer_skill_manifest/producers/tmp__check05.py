import sys, pickle
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; n=len(x)
rng=np.random.default_rng(7071)
partitions=[('ordered',np.arange(n))]+[('random',rng.permutation(n)) for _ in range(25)]
for j in range(4):
 ix=np.argsort(x[:,j]); partitions.extend([('edge',ix),('edge',ix[::-1])])
rows={}
for tag,idx in partitions:
 i,j=idx[:52],idx[52:]; train={k: v[i] for k,v in t.items() if k!='summary'}; tr=train['traj']; yy=train['y']; train['summary']=np.array([yy.mean(),yy.std(),tr[:,-1].mean(),tr[:,-1].std()]); out=solve_batch(dict(s,train=train,test={'x':x[j]})); e=np.abs(out['y_hat']-t['y'][j]); q=np.abs(out['state_pred']-t['traj'][j]); rows.setdefault(tag,[]).append([e.mean(),q.mean(),q[:,-1].mean(),*q.mean(0)])
for tag,a in rows.items(): print(tag,'mean',np.round(np.mean(a,0),6),'max',np.round(np.max(a,0),6))
assert rows['ordered'][0][0]<=.038
