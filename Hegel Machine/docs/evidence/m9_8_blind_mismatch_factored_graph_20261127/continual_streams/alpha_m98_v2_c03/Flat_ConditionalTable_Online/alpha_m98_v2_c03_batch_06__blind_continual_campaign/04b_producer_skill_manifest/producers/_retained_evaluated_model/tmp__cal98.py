import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; print('cal n, ranges',X.shape,np.round(np.quantile(X,[0,.5,1],axis=0),3),np.round(np.quantile(Y,[0,.5,1],axis=0),3)); print('first rows',np.round(np.c_[X,Y][:8],3)); print('single sweeps',C['single_channel_sweeps']);
def basis(x,kind):
 a=[np.ones(len(x)),*x.T]
 if kind=='diag': a.extend((x*x).T)
 if kind in ('quad','cube'): a.extend(x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2))
 if kind=='cube': a.extend(x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations_with_replacement(range(4),3))
 return np.column_stack(a)
rng=np.random.default_rng(98); folds=np.array_split(rng.permutation(len(X)),5)
for x,y,direction in ((Y,X,'inverse'),(X,Y,'forward')):
 for k in ('linear','diag','quad','cube'):
  A=basis(x,k)
  for robust in ('linear','soft_l1','cauchy'):
   p=np.empty_like(y)
   for ii in folds:
    mask=~np.isin(np.arange(len(x)),ii); a=A[mask]; b=y[mask]; q=np.linalg.lstsq(a,b,rcond=None)[0]; sc=max(.00001,np.median(abs(a@q-b)))
    if robust!='linear': q=np.column_stack([least_squares(lambda z:(a@z-bi)/sc, qi,loss=robust,x_scale='jac').x for bi,qi in zip(b.T,q.T)])
    p[ii]=A[ii]@q
   r=p-y; print(direction,k,robust,'channel/total',np.round(np.r_[np.quantile(abs(r),[.5,.9,.99]),np.quantile(abs(r.sum(1)),[.5,.9,.99])],5))
