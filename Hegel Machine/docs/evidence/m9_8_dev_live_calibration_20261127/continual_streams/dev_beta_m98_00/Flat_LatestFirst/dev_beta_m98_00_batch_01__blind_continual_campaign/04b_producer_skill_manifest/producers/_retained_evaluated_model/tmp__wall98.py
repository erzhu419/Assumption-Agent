from probe98 import *
from itertools import product
scale=np.array([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']])
for mask in product([0,1],repeat=2):
 q=np.zeros(12); q[[0,2]]=-mask[0]*scale[0]; q[[6,8]]=-mask[1]*scale[1]
 err=(pred(q)-Y).reshape(len(s['mu_shifts']),-1,Y.shape[1]); print('WALL_CANDIDATE',mask,'shift/pulse rmse',np.sqrt(np.mean(err**2,axis=1)).tolist())
for pulse in range(W.shape[1]):
 def params(f):
  q=np.zeros(12); q[[0,2]]=-f[0]*scale[0]; q[[6,8]]=-f[1]*scale[1]; return q
 fit=least_squares(lambda f:(pred(params(f))-Y)[:,pulse],np.zeros(2),gtol=1e-13)
 err=(pred(params(fit.x))-Y)[:,1-pulse]; print('HELD_OUT',1-pulse,'factors',fit.x,'rmse',np.sqrt(np.mean(err**2)))
for model in ('diagonal','affine'):
 es=[]
 for ch in range(4):
  x=basis(ref,1) if model=='affine' else np.column_stack([np.ones(len(ref)),ref[:,ch]])
  q=np.zeros(len(ref))
  for j in range(5):
   m=np.arange(len(ref))%5==j; q[m]=x[m]@np.linalg.lstsq(x[~m],raw[~m,ch],rcond=None)[0]-raw[m,ch]
  es.append(np.sqrt(np.mean(q*q)))
 print('SENSOR_CV',model,es)
