import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task');from batch_solver import *;from batch_solver import _panels,_scalar_parts
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x,y,tr=t['x'],t['y'],t['traj'];
I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]]
for i in I:
 j=np.setdiff1d(np.arange(72),i); d={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':x[j]}}; p=solve_batch(d); vs=_scalar_parts(_panels(s,x[np.r_[i,j]]),y[i]);print('CV',np.mean(abs(p['y_hat']-y[j])),np.mean(abs(p['state_pred']-tr[j])),np.mean(abs(vs-y[j]),axis=1));
 if len(i)==52:print('largest',np.c_[x[j],y[j],p['y_hat']][np.argsort(abs(p['y_hat']-y[j]))[-8:]])
print('x train slices/test mean',x[:52].mean(0),x[52:].mean(0),s['test']['x'].mean(0));print('y slice scale',np.std(y[:52]),np.std(y[52:]));
