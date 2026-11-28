import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u'];v=s['panel_a']['v'];np.set_printoptions(precision=3,suppress=True,linewidth=140)
for name,D in [('aff',np.c_[np.ones(len(u)),u]),('diag3',np.c_[np.ones(len(u)),u,u*u,u**3]),('cyc3',np.c_[np.ones(len(u)),u,u*np.roll(u,-1,axis=1),u**3]),('full2',np.c_[np.ones(len(u)),u,np.stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)],1)]),('full2diag3',np.c_[np.ones(len(u)),u,np.stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)],1),u**3])]:
 rng=np.random.default_rng(42); best=(1e9,None)
 for j in range(1000):
  ix=rng.choice(len(u),len(D.T)+2,replace=False);c=np.linalg.lstsq(D[ix],v[ix],rcond=None)[0];
  for _ in range(3):
   r=np.linalg.norm(D@c-v,axis=1); ix=np.argsort(r)[:38];c=np.linalg.lstsq(D[ix],v[ix],rcond=None)[0]
  r=np.linalg.norm(D@c-v,axis=1); e=np.sort(r)[:38].mean();
  if e<best[0]:best=(e,c)
 c=best[1];r=np.linalg.norm(D@c-v,axis=1);print(name,best[0],np.sort(r),c,sep='\n')
