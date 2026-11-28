import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; t=s['train']; ids=np.arange(len(t['x']))
a=m._calibrate(s['panel_a'],np.vstack([t['x'],s['test']['x']]))
c=m._operators(s,a)
sp=m._design(a,c,(a*c).sum(1),a*np.roll(c,1,axis=1)); h=sp[:len(ids)]; inv=np.linalg.pinv(h.T@h)
lev=np.einsum('ij,jk,ik->i',sp,inv,sp)
print('spatial leverage train and test maxima',lev[:len(ids)].max(), lev[len(ids):].max())
folds={'required':ids[52:],'gate':ids[56:]}
folds.update({f'block{i}':ids[i:i+18] for i in range(0,72,18)})
folds['high_activation']=np.argsort(np.linalg.norm(c[:len(ids)],axis=1))[-18:]
folds['high_leverage']=np.argsort(lev[:len(ids)])[-18:]
for j in range(4):
 q=np.argsort(a[:len(ids),j]); folds[f'axis{j}_tails']=np.r_[q[:8],q[-8:]]
results=[]
for name,j in folds.items():
 i=np.setdiff1d(ids,j); st=dict(s,train={**t,**{k:t[k][i] for k in ('x','y','traj')}},test={'x':t['x'][j]})
 out=m.solve_batch(st); ey=abs(out['y_hat']-t['y'][j]); et=abs(out['state_pred']-t['traj'][j]); e=[ey.mean(),et.mean(),et[:,-1].mean(),et.mean(0).max()]; results.append(e)
 print(name, np.round(e,6))
 assert max(e) <= .038, (name,e)
print('worst y/traj/terminal/timestamp',np.max(results,axis=0))
