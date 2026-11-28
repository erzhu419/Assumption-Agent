import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; np.set_printoptions(precision=5,suppress=True)
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
funs={'sin':lambda z,k:np.sin(k*z),'tanh':lambda z,k:np.tanh(k*z),'atan':lambda z,k:np.arctan(k*z),'rat':lambda z,k:z/(1+k*z*z),'cube':lambda z,k:z**3/(1+k*z*z),'sinh':lambda z,k:np.sinh(k*z),'erf':lambda z,k:__import__('scipy').special.erf(k*z)}
for name,f in funs.items():
 for extra in ('none','squares','cyclic','shift','cubes'):
  def m(p,z):
   q=z
   if extra=='shift': q=z+p[1:]
   cols=[np.ones(len(z)),z,f(q,p[0])];
   if extra=='squares': cols.append(z*z)
   if extra=='cyclic': cols.append(z*np.roll(z,-1,axis=1))
   if extra=='cubes': cols.append(z**3)
   return np.column_stack(cols)
  def err(p,a,b):
   mat=m(p,u[a]); return m(p,u[b])@ls(mat,v[a])-v[b]
  init={'sin':2.9,'tanh':1.45,'atan':1.35,'rat':.85,'cube':.85,'sinh':.8,'erf':1.2}[name]
  rms=[]; ma=[]; i=np.arange(len(u));
  for fold in range(5):
   a=i%4!=fold if fold<4 else i>=0; b=~a if fold<4 else a
   p=least_squares(lambda p:err(p,a,a).ravel(),np.r_[init,np.zeros(4)] if extra=='shift' else [init],bounds=(np.r_[.02,[-.5]*4] if extra=='shift' else [.02],np.r_[5.,[.5]*4] if extra=='shift' else [5.]),max_nfev=100).x
   r=err(p,a,b); ma.append(np.mean(abs(r)));rms.append(np.sqrt(np.mean(r*r)))
  print(name,extra,'cv avg/max/train',np.round([np.mean(ma[:4]),max(ma[:4]),ma[-1]],6),'p',np.round(p,5));
  if extra=='none':
   print('coef',ls(m(p,u),v).T)
   r=err(p,a,a); print('resid singular',np.linalg.svd(r,compute_uv=False))
