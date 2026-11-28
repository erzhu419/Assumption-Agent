from p import *
from scipy.linalg import solve
from scipy.optimize import minimize
from math import factorial

def kernels(a,b):
 g=a@b.T
 return [np.ones_like(g),g,g*g,g**3,g**4,g**5,g**6]

def search(ii,j,kind):
 a=x[ii]; b=Y[ii,j]; b=b.reshape(len(ii),-1); K=kernels(a,a); sc=np.std(b,axis=0)
 def model(h,K):
  if kind==0: return sum(q*np.exp(w) for q,w in zip(K,h[:-1]))
  if kind==1: return (1+np.exp(h[0])*K[1])**int(h[1])
 def fun(h):
  M=model(h,K)+np.eye(len(ii))*np.exp(h[-1]); inv=solve(M,np.eye(len(ii)),assume_a='pos'); e=(inv@b)/np.diag(inv)[:,None]/sc
  return np.mean(e**2)
 if kind==0: h=minimize(fun,[1,0,0,-2,-4,-5,-6,-7],bounds=[(-15,8)]*8,method='L-BFGS-B').x
 else: h=min((minimize(fun,[0,o,-7],bounds=[(-7,3),(o,o),(-15,4)],method='L-BFGS-B').x for o in range(2,8)),key=fun)
 return h,lambda jj:model(h,kernels(x[jj],a))@solve(model(h,K)+np.eye(len(ii))*np.exp(h[-1]),b,assume_a='pos')
for j in (0,slice(1,None)):
 for kind in (0,1):
  folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
  err=[]
  for jj in folds:
   h,pred=search(np.setdiff1d(np.arange(72),jj),j,kind); err.append(np.mean(abs(Y[jj,j].reshape(len(jj),-1)-pred(jj))))
  print(j,kind,'errors',np.round(err,5),'params',h.round(2))

a=s['panel_c']['u']; V=s['panel_c'];
for key in ('v1','v2'):
 print('panel C',key)
 for j in range(4):
  b=V[key][:,j]; ids=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(a[:,c],axis=1) for c in ids]); co=np.linalg.lstsq(A,b,rcond=None)[0]; print(j,'range',np.round([min(b),max(b)],3),'quad rmse',np.std(b-A@co).round(4),[(i,round(co[i2],3)) for i2,i in enumerate(ids) if abs(co[i2])>0.12])
# Check whether nonlinear terms cancel between response and trajectory modes.
U,S,R=np.linalg.svd(T,full_matrices=False); W=U[:,:2]*S[:2]; Z=np.c_[y,W];
for k in (1,2,3):
 F=poly(k); e=Z-F@np.linalg.lstsq(F,Z,rcond=None)[0]; _,D,C=np.linalg.svd(e,full_matrices=False); print('target residual singular',k,D.round(5),C.round(3))
