import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _spatial,_time_basis,_fit,solve_batch
from probe import s,tr,t,np
from sklearn.model_selection import KFold
x=tr['x']; y=tr['y']; traj=tr['traj']; a=_spatial(s,x); h=_time_basis(t,traj); tar=np.c_[y,_fit(h,traj.T).T]
labels=['offset']+[f'{k}{i}' for k in ('z','f','g') for i in range(4)]+[f'z{i}z{j}' for i in range(4) for j in range(i,4)]+[f'fg{roll}:{i}' for roll in (0,1,-1) for i in range(4)]
c=np.linalg.solve(a.T@a+.001*np.eye(a.shape[1]),a.T@tar)
for name,k in zip(labels,c): print(name,k.round(4))
z=(x-np.median(x,axis=0))/x.std(0); order=np.argsort(np.max(abs(z),axis=1))
splits=[('52/20',np.arange(52),np.arange(52,72)),('56/16',np.arange(56),np.arange(56,72)),('boundary',order[:-16],order[-16:])]+[(f'fold{i}',a,b) for i,(a,b) in enumerate(KFold(5,shuffle=True,random_state=104).split(x))]
for tag,u,v in splits:
 p=solve_batch(dict(s,train=dict(x=x[u],y=y[u],traj=traj[u]),test=dict(x=x[v]))); e=abs(p['state_pred']-traj[v]);ey=abs(p['y_hat']-y[v]);print(tag,'y',ey.mean(),'traj',e.mean(),'worst_time',e.mean(0).max(),'max y residual',ey.max())
