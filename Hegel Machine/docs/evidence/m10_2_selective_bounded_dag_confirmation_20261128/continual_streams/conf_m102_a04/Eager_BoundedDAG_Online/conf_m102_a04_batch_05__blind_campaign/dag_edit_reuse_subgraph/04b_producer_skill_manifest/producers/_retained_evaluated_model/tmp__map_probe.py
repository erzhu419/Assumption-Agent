import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align,_panel_maps
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
a=s['panel_a']; u=a['u']; v=a['v']; o,b=_align(a); r=v-o-b*u
print('affine calibration',o,b);print('calibration ordered residual',np.sort(abs(r),axis=0)[::4]);print('panel B residual affine',np.sort(abs(s['panel_b']['v']-o-b*s['panel_b']['u']),axis=0)[::4]);
c=s['panel_c']; u=c['u']; print('archived C rmse',np.sqrt(np.mean((_panel_maps(c,u)-np.c_[c['v1'],c['v2']])**2,axis=0)))
one=np.ones_like(u[:,0]); q=u*u
bases={'lin':[u], 'quadratic_diag':[u,q], 'cubic_diag':[u,q,u*q], 'quadratic_all':[u,q]+[u[:,i:i+1]*u[:,j:j+1] for i in range(4) for j in range(i)], 'softsign':[u/(1+abs(u))], 'tanh':[np.tanh(u)], 'asinh':[np.arcsinh(u)], 'sin':[np.sin(u)], 'smoothsat':[u/np.sqrt(1+q)], 'oddpoly':[u,u*q,u*q*q], 'rationalquad':[u,q,q/(1+.3*q)]}
for k in ('v1','v2'):
 y=c[k]; print(k)
 for name, f in bases.items():
  X=np.column_stack([one,*f]); P=np.linalg.pinv(X); B=P@y; r=X@B-y; loo=r/(1-np.sum(X*P.T,axis=1))[:,None]; print(name,'rmse',np.sqrt(np.mean(r*r)),'loo',np.sqrt(np.mean(loo*loo)))
  if name in ('lin','quadratic_diag','cubic_diag'): print(B)
print('training quantiles calibrated',np.quantile((s['train']['x']-o)/b,[0,.1,.5,.9,1],axis=0))
