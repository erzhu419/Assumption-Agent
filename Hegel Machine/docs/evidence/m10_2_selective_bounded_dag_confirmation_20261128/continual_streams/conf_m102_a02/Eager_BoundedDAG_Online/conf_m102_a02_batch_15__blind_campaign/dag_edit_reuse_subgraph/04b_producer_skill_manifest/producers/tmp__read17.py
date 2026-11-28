import numpy as np
from itertools import combinations_with_replacement
p=np.load('/tmp/features17.npz'); g,z,Y=p['g'],p['z'],p['targets']; A=np.column_stack([np.ones(len(g)),g]); names=list(combinations_with_replacement(range(4),2)); Q=np.column_stack([g[:,i]*g[:,j] for i,j in names]); rng=np.random.default_rng(311); masks=[(np.arange(52),np.arange(52,72))]; masks += [(a[:52],a[52:]) for a in [rng.permutation(72) for _ in range(32)]]; masks += [(np.argsort(np.linalg.norm(z,axis=1))[:-15],np.argsort(np.linalg.norm(z,axis=1))[-15:])];
for name,B in [('affine',A),('opposite',np.column_stack([A,Q[:,[names.index((0,2)),names.index((1,3))]]])),('full',np.column_stack([A,Q]))]:
 for col in [slice(0,1),slice(1,9)]:
  errs=[]; cs=[]
  for i,j in masks:
   t=Y[i,col]; c=np.linalg.lstsq(B[i],t,rcond=None)[0]; cs.append(c); errs.append(np.abs(B[j]@c-Y[j,col]).mean())
  print(name,col,errs[0],max(errs),errs[-1]); print('coef average',np.round(np.mean(cs,axis=0),4) if col.start==0 else np.round(np.linalg.norm(np.mean(cs,axis=0),axis=1),4))
