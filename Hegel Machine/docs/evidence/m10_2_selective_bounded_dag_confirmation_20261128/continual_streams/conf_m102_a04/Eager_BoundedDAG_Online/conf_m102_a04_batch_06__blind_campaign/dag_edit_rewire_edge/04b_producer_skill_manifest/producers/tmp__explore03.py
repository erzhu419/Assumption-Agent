import sys,pickle,numpy as np, importlib.util
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_01 as b,node_02 as c
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; y=np.c_[tr['y'],tr['traj']]; N=len(y)
z=a._calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']]); print('z range',z.min(0),z.max(0)); print('z abs maxima train rows',np.argsort(np.max(abs(z[:N]),axis=1))[-10:]); print('targets range',y.min(0),y.max(0)); print('traj singular values',np.linalg.svd(y[:,1:],compute_uv=False))
for n in (48,50,54,56,60):
 t=s.copy(); t['train']={k:v[:n] for k,v in tr.items() if k!='summary'};t['test']={'x':tr['x'][n:]}; d=a.run_node({'__input__':t}); e=b.run_node({'__input__':t}); f=c.run_node({'node_00':d,'node_01':e})['fit']; w=d['features']; print('split',n,'features abs max',np.max(abs(w),axis=0))
 def check(name,x):
  g=abs(x-y[n:]); print(name,np.round([g[:,0].mean(),g[:,1:].mean(),g[:,0].max(),g[:,1:].mean(1).max()],5))
 check('frozen',f)
 for name,v in [('w',w),('w1',w[:,:4]),('w2',w[:,4:]),('z',z[:N]),('even',np.c_[z[:N]**2,z[:N]*np.roll(z[:N],-1,axis=1)])]:
  for deg in (1,2,3):
   if deg==3 and v.shape[1]>4:continue
   X=np.column_stack([np.ones(N)]+[np.prod(v[:,inds],axis=1) for h in range(1,deg+1) for inds in combinations_with_replacement(range(v.shape[1]),h)])
   check(name+str(deg),X[n:]@np.linalg.lstsq(X[:n],y[:n],rcond=1e-9)[0])
