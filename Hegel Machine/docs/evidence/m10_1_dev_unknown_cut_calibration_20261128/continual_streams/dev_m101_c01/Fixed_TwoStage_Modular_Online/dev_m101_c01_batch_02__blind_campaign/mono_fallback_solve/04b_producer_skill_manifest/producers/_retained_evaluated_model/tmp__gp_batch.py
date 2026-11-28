import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement

def polys(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for j in combinations_with_replacement(range(4),k)])

def gp(x,y,q,kind):
 xq=np.r_[x,q]; n=len(x); xq=xq/.7
 y=np.asarray(y); y=y[:,None] if y.ndim==1 else y
 y0=y.mean(0); sy=np.maximum(y.std(0),1e-8); ymm=(y-y0)/sy
 Q=[polys(xq,k) for k in range(1,5)]; W=[p@p.T for p in Q]; D=(xq[:,None,:]-xq[None,:,:])**2; Dtr=D[:n,:n]; eye=np.eye(n)
 trW=[w[:n,:n] for w in W]; const=np.ones((len(xq),len(xq)))
 if kind=='poly': init=[-.3,-2,-4,-6,-8]; bound=[(-9,2)]*4+[(-13,-3)]
 else: init=[-.3,-2,-3,0,0,0,0,-8]; bound=[(-9,2)]*3+[(-1.8,2)]*4+[(-13,-3)]
 def matrices(p,train):
  a=np.exp(p); w=trW if train else W; c=const[:n,:n] if train else const[n:,:n]; w=w if train else [b[n:,:n] for b in w]
  if kind=='poly':
   gg=[a[i]*w[i] for i in range(4)]; k=c+sum(gg)
  else:
   delta=Dtr if train else D[n:,:n]; dt=delta*a[3:7]; dist=dt.sum(-1); g=np.exp(-.5*dist)
   if kind=='rq': g=(1+.5*dist)**-1
   a0,a1,a2=a[:3]; k=c+a0*w[0]+a1*w[1]+a2*g; gg=[a0*w[0],a1*w[1],a2*g]; gg.extend(np.moveaxis(-.5*a2*g[:,:,None]*dt*(g[:,:,None] if kind=='rq' else 1),2,0))
  if train: k=k+a[-1]*eye; gg.append(a[-1]*eye)
  return k,gg
 def loss(p):
  k,g=matrices(p,True); fac=cho_factor(k,lower=True,check_finite=False); c=cho_solve(fac,ymm,check_finite=False); inv=cho_solve(fac,eye,check_finite=False); J=y.shape[1]*inv-c@c.T
  prior=np.zeros(len(p)); prior[3:7]=.10 if kind!='poly' else 0
  return .5*np.sum(c*ymm)+y.shape[1]*np.log(np.diag(fac[0])).sum()+np.sum(prior*p*p), .5*np.einsum('ij,kij->k',J,g)+2*prior*p
 fit=minimize(loss,init,jac=True,method='L-BFGS-B',bounds=bound,options={'maxiter':160,'ftol':1e-10}); k,_=matrices(fit.x,True); cross,_=matrices(fit.x,False)
 return y0+sy*(cross@cho_solve(cho_factor(k,lower=True),ymm)),fit.x

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=d['x'],d['y'],d['traj']; n=len(x)
 folds=[np.arange(56,72),*np.array_split(np.random.default_rng(22).permutation(n),5), np.argsort(np.linalg.norm(x,axis=1))[-14:]]
 for kind in ['poly','rbf','rq']:
  vals=[]
  for j in folds:
   i=np.setdiff1d(np.arange(n),j); V=np.linalg.svd(z[i],full_matrices=False)[2][:2]; a=z[i]@V.T
   yy,yp=gp(x[i],y[i],x[j],kind); aa,ap=gp(x[i],a,x[j],kind); ac=np.c_[*[gp(x[i],col,x[j],kind)[0][:,0] for col in a.T]]
   vals.append([abs(yy[:,0]-y[j]).mean(),abs(aa@V-z[j]).mean(),abs(ac@V-z[j]).mean()]);
   if len(vals)==1:print(kind,'parameters',yp.round(2),ap.round(2))
  print(kind,np.round(vals,5))
