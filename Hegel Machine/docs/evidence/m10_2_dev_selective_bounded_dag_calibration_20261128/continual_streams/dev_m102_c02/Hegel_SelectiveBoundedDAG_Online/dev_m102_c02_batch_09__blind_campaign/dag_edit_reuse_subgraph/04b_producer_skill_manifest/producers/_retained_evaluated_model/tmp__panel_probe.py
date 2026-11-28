import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=170)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def errors(a,b):
 p=a@ls(a,b); e=np.linalg.norm(p-b,axis=1); return np.quantile(e,[0,.25,.5,.75,.9,1])
for panel in ('panel_a','panel_b','panel_c'):
 u=s[panel]['u']; print(panel)
 for key,v in s[panel].items():
  if key=='u': continue
  for degree in range(1,5):
   diag=np.c_[np.ones(len(u)),*[u**j for j in range(1,degree+1)]]
   poly=np.c_[np.ones(len(u)),*[np.prod(u[:,idx],axis=1) for d in range(1,degree+1) for idx in combinations_with_replacement(range(4),d)]]
   print(key,degree,'diag',errors(diag,v),'full',errors(poly,v))
  if panel=='panel_a':
   print('rows: index, input, measured, affine residual norm'); e=np.linalg.norm(np.c_[np.ones(len(u)),u]@ls(np.c_[np.ones(len(u)),u],v)-v,axis=1)
   print(np.c_[np.arange(len(u)),u,v,e])
