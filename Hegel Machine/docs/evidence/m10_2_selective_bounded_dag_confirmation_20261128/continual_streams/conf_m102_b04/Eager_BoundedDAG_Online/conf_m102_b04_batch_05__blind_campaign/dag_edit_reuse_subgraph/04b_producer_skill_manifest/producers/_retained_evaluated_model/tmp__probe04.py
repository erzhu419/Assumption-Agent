import pickle,sys,numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=run_node({'__input__':s}); n=len(s['train']['x']); x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; q=out['hybrid'][:n,36:40]
np.set_printoptions(precision=4,suppress=True,linewidth=160)
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(x.shape[1]),k)])
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u'];
 for k,v in s[p].items():
  if k=='u':continue
  f=poly(u,1); fit=np.linalg.lstsq(f,v,rcond=None)[0]; print(p,k,'fit',fit,'res',np.sort(np.linalg.norm(v-f@fit,axis=1)), 'sv',np.linalg.svd(fit[1:],compute_uv=False));
print('calib ranges',q.min(0),q.max(0));print('time sv',np.linalg.svd(z,full_matrices=False)[1]);print('time vec',np.linalg.svd(z,full_matrices=False)[2][:5]);print('summary',s['train']['summary'])
rng=np.random.default_rng(77); perm=rng.permutation(n); folds=np.array_split(perm,4)
def cv(f,b,reg,k=None):
 pred=np.empty_like(b)
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); g=f[tr]; h=b[tr];
  if k: _,_,v=np.linalg.svd(h,full_matrices=False); h=h@v[:k].T
  coef=np.linalg.solve(g.T@g+reg*np.eye(g.shape[1]),g.T@h)
  pred[te]=f[te]@coef if not k else (f[te]@coef)@v[:k]
 e=np.abs(pred-b);return tuple(round(v,5) for v in [e.mean(),max(e[i].mean() for i in folds)])
for name,f in list((k,v[:n]) for k,v in out.items())+[(f'poly_{d}_{h}',poly(q/np.sqrt(1+h*q*q),d)) for d in (1,2,3,4) for h in (0,.09,.3,.7,1.25)]:
 print(name, 'y', min((cv(f,y,r),r) for r in (.000001,.0001,.001,.007,.03,.1,1)), 'z',min((cv(f,z,r),r) for r in (.000001,.0001,.001,.007,.03,.1,1)))
