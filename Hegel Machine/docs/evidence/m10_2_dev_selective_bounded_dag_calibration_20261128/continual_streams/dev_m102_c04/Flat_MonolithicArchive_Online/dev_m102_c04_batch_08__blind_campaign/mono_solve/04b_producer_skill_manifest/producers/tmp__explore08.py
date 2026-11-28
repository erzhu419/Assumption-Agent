import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda X,Y:np.linalg.lstsq(X,Y,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def powers(x,n):return np.stack([np.prod(x[:,i],axis=1) for k in range(1,n+1) for i in itertools.combinations_with_replacement(range(4),k)],axis=1)
u,v=[s['panel_a'][k] for k in ('u','v')]; I=np.ones(len(u),bool)
for _ in range(9):
 w=ls(aff(u)[I],v[I]); err=np.linalg.norm(aff(u)@w-v,axis=1); I=err<4*np.median(err)
print('a indices, affine, residual:',I.sum(),w,np.sort(err))
for degree in (2,3):
 a=aff(powers(u,degree));w=ls(a[I],v[I]); print('a poly',degree,np.sqrt(np.mean((a[I]@w-v[I])**2)));print(np.round(w,3))
u=s['panel_c']['u']
a=aff(powers(u,2))
print('c1 coeffs',np.round(ls(a,s['panel_c']['v1']),4))
v=s['panel_c']['v2']
for base in (aff(u)[:,:1],aff(u),a):
 for name,fn in [('cos',lambda u,k:np.cos(k*u)),('sin',lambda u,k:np.sin(k*u)),('tanh',lambda u,k:np.tanh(k*u)),('sinc',lambda u,k:np.sinc(k*u)),('rat1',lambda u,k:u/(1+k*abs(u))),('rat2',lambda u,k:u*u/(1+k*u*u)),('exp',lambda u,k:np.exp(-k*u*u)),('cos_sin',lambda u,k:np.c_[np.cos(k*u),np.sin(k*u)])]:
  def err(k):
   z=np.c_[base,fn(u,k)]; w=ls(z,v); return np.sqrt(np.mean((z@w-v)**2))
  g=np.linspace(.04,6,120); j=np.argmin([err(k) for k in g]); r=minimize_scalar(err,bounds=(max(.001,g[j]-.055),g[j]+.055),method='bounded');print('c2',base.shape[1],name,round(r.x,6),round(r.fun,7))
