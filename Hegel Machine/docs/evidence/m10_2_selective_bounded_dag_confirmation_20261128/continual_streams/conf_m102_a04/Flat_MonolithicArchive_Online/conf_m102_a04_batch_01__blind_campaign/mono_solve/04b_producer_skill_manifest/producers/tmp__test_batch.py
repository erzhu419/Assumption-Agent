import pickle, numpy as np, sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ix=np.arange(72)
Splits=[ix[:52]]+[np.random.default_rng(i).permutation(ix)[:52] for i in range(40)]
for j in np.array_split(np.argsort(np.linalg.norm(tr['x'],axis=1)),4): Splits.append(np.setdiff1d(ix,j))
A=[]
for train in Splits:
 test=np.setdiff1d(ix,train); st=dict(s,train={k:v[train] if k!='summary' else v for k,v in tr.items()},test={'x':tr['x'][test]}); p=m.solve_batch(st)
 e=abs(p['state_pred']-tr['traj'][test]); A.append(np.r_[abs(p['y_hat']-tr['y'][test]).mean(),e.mean(),e.mean(0)])
A=np.array(A); print('prescribed',A[0]); print('random worst',A[1:41].max(0)); print('norm slices',A[-4:])
