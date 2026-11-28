import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
import node_00 as a,node_01 as b,node_02 as c,node_03 as d
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; y=np.c_[t['y'],t['traj']]; z=a._calibrate(s['panel_a'],t['x']); rng=np.random.default_rng(2026)
for train,test in [(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(6)]]:
 st=dict(s,train={k:t[k][train] for k in ('x','y','traj')},test={'x':t['x'][test]}); f=a.run_node({'__input__':st}); g=b.run_node({'__input__':st}); h=c.run_node({'node_00':f,'node_01':g}); j=d.run_node({'node_00':f,'node_01':g,'node_02':h}); err=np.abs(np.c_[j['y_hat'],j['state_pred']]-y[test]); print('candidate',*(round(v,5) for v in (err[:,0].mean(),err[:,1:].mean(),err[:,0].max(),err[:,1:].mean(1).max())))
 # Diagnostic only: compare the same kernel with and without access to the sign.
 D=cdist(z,z,'sqeuclidean'); A=cdist(z,-z,'sqeuclidean'); K=np.exp(-.6*D); V=np.exp(-.6*A); E=(K+V)/2; O=(K-V)/2
 for name,M in [('even',E),('signed',E+O)]:
  M=M+1; u=solve(M[np.ix_(train,train)]+.07*np.eye(len(train)),y[train],assume_a='pos'); e=np.abs(M[np.ix_(test,train)]@u-y[test]); print(name,round(e[:,0].mean(),5),round(e[:,1:].mean(),5))
print('exact sign-collision deltas',np.max(np.abs(a._interactions(z)-a._interactions(-z))),np.max(np.abs(a._curvature(z,.25)-a._curvature(-z,.25))))
