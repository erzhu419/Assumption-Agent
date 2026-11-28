import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations

def ard(x,y,kind=0,detail=False):
 n=len(y); m=len(x[0]); mu=y.mean(); sd=max(y.std(),1e-8); yy=(y-mu)/sd
 x0=x[0]; mats=[np.ones((m,n))]; names=['const']
 def add(k,name):
  sc=np.mean(k[np.arange(n),np.arange(n)]); mats.append(k/max(sc,1.e-10)); names.append(name)
 def feat(z,name):
  z=z-z[:n].mean(); add(np.outer(z,z[:n]),name)
 for i,q in enumerate(x):
  q=q-q[:n].mean(0); q=q/np.maximum(q[:n].std(0),1e-8); g=q@q[:n].T/q.shape[1]; dd=(q[:,None,:]-q[None,:n,:])**2
  for p in [1,2,3]:add(g**p,(i,'poly',p))
  for a in [.07,.35,1.4]:add(np.exp(-a*dd.mean(2)),(i,'rbf',a))
  for j,z in enumerate(q.T):
   for nm,fn in [('lin',lambda z:z),('sq',lambda z:z*z),('sat',lambda z:np.arcsinh(1.7*z)),('sin',lambda z:np.sin(1.15*z)),('cube',lambda z:z**3)]:feat(fn(z),(i,nm,j))
   for a in [.25,1.]:add(np.exp(-a*dd[:,:,j]),(i,'add',j,a))
  for j,k in combinations(range(q.shape[1]),2):
   z,h=q[:,j],q[:,k]; feat(z*h,(i,'prod',j,k));feat(z*h/(1+.22*(z*z+h*h)),(i,'rat',j,k)); add(np.exp(-.2*(dd[:,:,j]+dd[:,:,k])),(i,'pair',j,k))
 M=np.array(mats); C=np.concatenate([M[:,:n,:],np.eye(n)[None]]).reshape(len(M)+1,-1); eye=np.eye(n); pen=[.035,.10,.008][kind]; center=np.r_[np.zeros(1),np.full(len(M)-1,-5.),-9.]
 def fun(p):
  a=np.exp(p); fac=cho_factor((a@C).reshape(n,n),check_finite=False); al=cho_solve(fac,yy,check_finite=False); iv=cho_solve(fac,eye,check_finite=False); grad=(iv-np.outer(al,al))*.5
  return .5*yy@al+np.log(fac[0].diagonal()).sum()+pen*np.sum((p-center)**2), a*(C@grad.ravel())+2*pen*(p-center)
 p=minimize(fun,center,method='L-BFGS-B',jac=True,bounds=[(-16,4)]*len(M)+[(-16,-2)],options={'maxiter':160,'ftol':2e-8}).x
 a=np.exp(p); fac=cho_factor((a@C).reshape(n,n)); result=mu+sd*(a[:-1]@M[:,n:].reshape(len(M),-1)).reshape(m-n,n)@cho_solve(fac,yy)
 if detail:print([(names[j],round(a[j],4)) for j in np.argsort(a[:-1])[-12:][::-1]],'noise',a[-1]);return result
 return result
