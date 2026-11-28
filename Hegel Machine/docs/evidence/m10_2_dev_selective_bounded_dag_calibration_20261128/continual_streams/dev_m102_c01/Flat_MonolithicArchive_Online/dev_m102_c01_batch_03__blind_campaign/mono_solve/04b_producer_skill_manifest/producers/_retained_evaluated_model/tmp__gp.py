import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def gp(x,q,y,kind=0,seed=0,extra=None):
 y=np.asarray(y).reshape(len(x),-1); scale=y.std(0); ym=y.mean(0); Y=(y-ym)/scale
 d=x.shape[1]; i,j=np.triu_indices(d); DD=x[:,None]-x[None,:]; J=np.eye(len(x)); XL=np.c_[np.ones(len(x)),x]; L=XL@XL.T
 B=np.zeros((len(i),d,d)); B[np.arange(len(i)),i,j]=1
 if kind in (1,3): B=B[i==j]
 if kind in (4,5): B=np.zeros((0,d,d))
 mats=[DD[:,:,a]*DD[:,:,b] for a in range(d) for b in range(d)]; mats=np.array(mats).reshape(d,d,-1)
 order=1 if kind in (2,3,5) else 0
 def calc(h,X=x,Z=x,grad=False):
  A=h[4] * np.eye(d)+np.einsum('a,aij->ij',h[5:],B); G=A@A.T; W=X[:,None]-Z[None,:]; D=np.einsum('...i,ij,...j->...',W,G,W); R=np.exp(-.5*D) if not order else (1+np.sqrt(3*D))*np.exp(-np.sqrt(3*D)); v=np.exp(h[:4]); lin=1+X@Z.T; K=v[0]*R+v[1]*lin+v[2]*lin**2
  if not grad:return K
  K+=v[3]*J; fac=cho_factor(K); a=cho_solve(fac,Y); val=.5*np.sum(Y*a)+Y.shape[1]*np.log(np.diag(fac[0])).sum(); Mi=.5*(Y.shape[1]*cho_solve(fac,J)-a@a.T); g0=[np.sum(Mi*v[0]*R),np.sum(Mi*v[1]*lin),np.sum(Mi*v[2]*lin**2),np.sum(Mi*v[3]*J)]; deriv=(-.5*R if not order else -1.5*np.exp(-np.sqrt(3*D)))*Mi*v[0]; S=(mats@deriv.ravel())@A; g=np.r_[g0,2*np.trace(S),2*np.einsum('ij,aij->a',S,B)];
  return val+.3*np.dot(h[5:],h[5:]),g+np.r_[np.zeros(5),.6*h[5:]]
 rng=np.random.default_rng(seed); opts=[]
 for k in range(2):
  h0=np.r_[0,-2,-5,-7,.7,np.zeros(len(B))]; h0[5:]=rng.normal(0,.2*k,len(B)); opts.append(minimize(lambda h:calc(h,grad=True),h0,jac=True,method='L-BFGS-B',bounds=[(-8,15),(-12,5),(-14,5),(-14,-1),(.01,5)]+[(-4,4)]*len(B), options={'maxiter':150,'ftol':1e-10}))
 opt=min(opts,key=lambda p:p.fun); H=opt.x; K=calc(H); K+=np.exp(H[3])*J; Q=calc(H,q,x); return (Q@cho_solve(cho_factor(K),Y))*scale+ym,H,opt.fun

if __name__=='__main__':
 import contextlib,io
 with contextlib.redirect_stdout(io.StringIO()): from pathprobe import *
 _,_,V=np.linalg.svd(t,full_matrices=False); T=t@V[:2].T
 for kind in range(6):
  for key in ['z','Bz','Iz'] if kind<4 else ['z']:
   q=D[key]; errs=[]
   for i,j in splits:
    h,par,l=gp(q[i],q[j],np.c_[y][i],kind); hh,pt,lt=gp(q[i],q[j],T[i],kind); errs.append([np.mean(abs(h[:,0]-y[j])),np.mean(abs(hh@V[:2]-t[j])),l]);
   print(kind,key,np.round(errs,5),flush=True)
 print('details')
 for i,j in splits[:1]:
  q=D['z']; h,p,l=gp(q[i],q[j],np.c_[y][i],0); hh,pt,lt=gp(q[i],q[j],T[i],0); print(p,pt); print(np.c_[q[j],y[j],h[:,0],y[j]-h[:,0]])
