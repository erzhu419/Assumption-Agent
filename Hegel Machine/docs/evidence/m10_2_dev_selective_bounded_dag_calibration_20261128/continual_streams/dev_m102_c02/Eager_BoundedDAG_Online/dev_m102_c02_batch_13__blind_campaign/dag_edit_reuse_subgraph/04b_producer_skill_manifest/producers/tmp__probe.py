import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def feats(u,d):
 return np.column_stack([np.ones(len(u))]+[np.prod(u[:,k],axis=1) for n in range(1,d+1) for k in combinations_with_replacement(range(u.shape[1]),n)])
for panel in ['panel_a','panel_b','panel_c']:
 u=s[panel]['u']; print('\n',panel)
 for k,v in s[panel].items():
  if k=='u':continue
  print(k,'affine correlations', np.corrcoef(u.T,v.T)[:4,4:]); print('sample',np.column_stack((u,v))[:7]);
  for d in [1,2,3]:
   X=feats(u,d); fit=np.linalg.lstsq(X,v,rcond=None)[0]; r=np.abs(X@fit-v); print(d,'MAE',r.mean(0),'LOO', np.mean(r/np.maximum(1-(X*np.linalg.pinv(X).T).sum(1),1e-10)[:,None],0))
  X=feats(u,1); fit=np.linalg.lstsq(X,v,rcond=None)[0]; print('affine coef',fit)
  if panel=='panel_a':
   for j in range(4):
    for d in [1,2,3]:
     for axis in [False,True]:
      X=feats(u[:,j:j+1] if axis else u,d); b=np.linalg.lstsq(X,v[:,j],rcond=None)[0]; p=least_squares(lambda b:X@b-v[:,j],b,loss='cauchy',f_scale=.003,max_nfev=1000).x; r=np.abs(X@p-v[:,j]); print(j,d,axis,np.quantile(r,[0,.25,.5,.75,.85,.9,1]), 'coef',p if d<=2 else '')
print('SVD traj',np.linalg.svd(s['train']['traj'],compute_uv=False))
