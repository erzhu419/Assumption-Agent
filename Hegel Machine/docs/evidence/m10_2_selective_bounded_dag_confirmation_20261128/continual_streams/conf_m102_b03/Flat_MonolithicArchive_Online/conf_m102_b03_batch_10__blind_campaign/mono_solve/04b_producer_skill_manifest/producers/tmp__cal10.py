import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; ls=lambda d,y:np.linalg.lstsq(d,y,rcond=1e-10)[0]; D=lambda x:np.c_[np.ones(len(x)),x]; rng=np.random.default_rng(15); A=D(u); best=min((ls(A[ix],v[ix]) for ix in (rng.choice(len(A),8,False) for _ in range(600))),key=lambda p:np.median(np.linalg.norm(A@p-v,axis=1))); r=np.linalg.norm(A@best-v,axis=1); print('robust residuals',np.sort(r)); good=r<min(2,6*np.median(r)); print('inliers',good.sum());
for j in range(4):
 for k in [(j+1)%4,(j-1)%4,None]:
  w=u[:,j] if k is None else u[:,k]; a=u[:,j]; P=D(np.c_[a,w,a*w,a**3]); c=ls(P[good],v[good,j]); print(j,k,np.std(P[good]@c-v[good,j]),c.round(6));
P=D(np.c_[u,np.column_stack([np.prod(u[:,ix],axis=1) for n in (2,3) for ix in combinations_with_replacement(range(4),n)])]); C=ls(P[good],v[good]); print('all polynomial',np.std(P[good]@C-v[good]),'max terms',[(str(ix),np.round(c,4)) for ix,c in zip([()]+[(i,) for i in range(4)]+[ix for n in (2,3) for ix in combinations_with_replacement(range(4),n)],C) if abs(c).max()>.002]);
