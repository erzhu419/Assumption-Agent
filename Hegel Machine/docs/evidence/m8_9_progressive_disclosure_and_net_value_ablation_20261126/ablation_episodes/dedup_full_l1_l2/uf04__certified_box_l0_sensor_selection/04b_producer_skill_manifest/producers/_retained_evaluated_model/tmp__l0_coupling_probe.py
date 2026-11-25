import importlib.util
import numpy as np
D=np.repeat(np.eye(8),2,axis=1)[:,:15]
for seed in (2,7,19,42,81):
 for c in (.003,.03,.12):
  X=D+c*np.random.default_rng(seed).normal(size=D.shape)
  a,b=X[:,:14:2],X[:,1:14:2]
  corr=np.sum(a*b,axis=0)/np.linalg.norm(a,axis=0)/np.linalg.norm(b,axis=0)
  print(seed,c,round(float(np.min(corr)),5),round(float(np.median(corr)),5))
