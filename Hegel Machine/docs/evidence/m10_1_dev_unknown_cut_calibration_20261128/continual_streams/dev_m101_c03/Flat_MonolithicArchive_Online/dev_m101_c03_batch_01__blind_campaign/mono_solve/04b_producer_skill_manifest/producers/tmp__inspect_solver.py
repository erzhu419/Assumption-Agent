import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; X=m._design(s,tr['x'])
np.set_printoptions(precision=5,suppress=True,linewidth=130)
K,A=m._kinetics(X,tr['traj'],t)
U=np.c_[A,tr['y']]; c=m._fit(X,U); err=X@c-U
se=(np.linalg.pinv(X.T@X).diagonal()[:,None]*np.mean(err**2,0))**.5
print('feature coefficients [slow, fast, y]:\n',c); print('standard errors:\n',se); print('coeff ratios to SE:\n',c/se); print('in-sample amplitude and y RMS',np.mean(err**2,0)**.5)
idx=np.arange(len(X)); splits=[idx[56:]]+list(np.array_split(np.random.default_rng(14).permutation(len(X)),4));
for J in splits:
 I=np.setdiff1d(idx,J); st=dict(s,train={**tr,**{k:tr[k][I] for k in ['x','y','traj']}},test={'x':tr['x'][J]}); o=m.solve_batch(st); e=np.abs(o['y_hat']-tr['y'][J]); q=np.abs(o['state_pred']-tr['traj'][J]);print('MAEs',e.mean(),q.mean(),'per time',q.mean(0))
