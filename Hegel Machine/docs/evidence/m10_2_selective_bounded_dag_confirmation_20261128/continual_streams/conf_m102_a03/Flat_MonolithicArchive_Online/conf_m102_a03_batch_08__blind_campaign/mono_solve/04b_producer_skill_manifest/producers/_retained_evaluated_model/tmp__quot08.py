exec(open('/tmp/explore08.py').read().split('for name in')[0])
from itertools import product
from scipy.optimize import minimize_scalar
h=m._warp(z,p); w=aff(h)@b; T=s['train']['traj']; n=len(y); rng=np.random.default_rng(42)
sp=[(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20))];
for _ in range(4):
 r=rng.permutation(n); sp.append((r[:52],r[52:]))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def profile(A,d,target,i,pen=.001,fixed=None):
 sc=np.sqrt(np.mean(A[i]**2,0)).clip(1e-8); AA=A/sc; I=np.eye(A.shape[1]);
 def fun(q,ret=False):
  denom=1+q*d*d; M=AA/denom[:,None]; a=M[i]; c=np.linalg.solve(a.T@a+pen*I,a.T@target[i]); yh=M@c
  return (yh,c/sc,q) if ret else np.sum((yh[i]-target[i])**2)+pen*np.sum(c*c)
 q=minimize_scalar(fun,bounds=(0,.75),method='bounded').x if fixed is None else fixed
 return fun(q,True)
for key,v in [('latent',h),('operator',w),('centered',w-b[0])]:
 for mode in ['bilin','quad']:
  A=np.c_[aff(z),v,cross(z,v)]
  if mode=='quad':A=np.c_[A,pol(v,2)[:,4:]]
  print(key,mode)
  for pen in [.0001,.004,.03,.15]:
   rr=[profile(A,z[:,3],y,i,pen) for i,j in sp]; print(pen,[(round(np.abs(a[j]-y[j]).mean(),5),round(q,4)) for (a,c,q),(i,j) in zip(rr,sp)])
  _,c,q=profile(A,z[:,3],y,np.arange(n)); print(q,c[:9].round(3),c[9:25].reshape(4,4).round(3),c[25:].round(3))
A=np.c_[aff(z),h,cross(z,h),pol(h,2)[:,4:]]
for si,(i,j) in enumerate(sp):
 for col in range(4):
  yh,c,q=profile(A,z[:,col],y,i,.004); print('coord',si,col,round(q,4),round(np.mean(np.abs(yh[i]-y[i])),5),round(np.mean(np.abs(yh[j]-y[j])),5))
 for X in [aff(h),np.c_[aff(h),cross(z,h)],A]:
  yp,c,q=profile(X,z[:,3],T,i,.015); base,_,_=profile(X,z[:,3],T,i,.015,0.); print('state',si,X.shape[1],round(q,4),round(np.abs(yp[j]-T[j]).mean(),5),round(np.abs(base[j]-T[j]).mean(),5))
