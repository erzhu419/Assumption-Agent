import pickle,itertools,runpy
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_00_solver.py'); np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,n):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(n+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; off,gain,m=old['_calibration'](s); print('old cal',off,gain); print('B gram',m[1:].T@m[1:]);
for X in (aff(u),poly(u,2),poly(u,3)):
 print('A basis',X.shape[1]);
 for t in range(4):
  w=least_squares(lambda w:X@w-v[:,t],ls(X,v[:,t]),loss='soft_l1',f_scale=.0005).x;e=abs(X@w-v[:,t]); keep=e<max(.002,4*np.median(e)); print(t,'median,max,n clean',np.median(e),max(e),sum(keep),'coef',w[:15]);
print('old A residual percentile',np.quantile(abs(u*gain+off-v),[0,.5,.75,.9,1],axis=0)); tr=s['train']; x=tr['x']; y=tr['y']; traj=tr['traj']; q=(x-off)/gain; z=aff(q)@m; p=s['panel_c'];
for k in ('v1','v2'):
 for X in (poly(p['u'],2),poly(p['u'],3),np.c_[aff(p['u']),p['u']**2,p['u']**3,np.sin(p['u']),np.cos(p['u'])]):
  w=ls(X,p[k]); print('C',k,X.shape[1],np.sqrt(np.mean((X@w-p[k])**2)), 'quadratic' if X.shape[1]==15 else '',w if X.shape[1]==15 else '')
 def unused():pass
print('stats',np.mean(traj[:,-1]),np.std(traj[:,-1]));
for name,X in [('linear',aff(z)),('old',np.c_[z,old['_cross'](z)]),('intercept cross',np.c_[aff(z),old['_cross'](z)]),('quadratic',poly(z,2)),('cubic',poly(z,3)),('diagonal quad',np.c_[aff(z),z**2]),('pairs',np.c_[aff(z),*[z[:,i]*z[:,j] for i,j in itertools.combinations(range(4),2)]])]:
 for Y in (y,traj):
  w=ls(X[:56],Y[:56]); print(name,Y.ndim,'train',np.mean(abs(X[:56]@w-Y[:56])),'test',np.mean(abs(X[56:]@w-Y[56:])),'coeff',w if Y.ndim==1 else w[:,-1])
