import numpy as np
from scipy.optimize import minimize_scalar,least_squares
J=[1,2,3,0]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def ones(u): return np.ones(len(u))
def a_fit(p):
 u=p['u']; v=p['v']; z=u[:,J]; X=np.stack([np.ones_like(u),u,z,u*z,u**3],axis=-1); W=np.ones(len(u));
 for _ in range(20):
  c=np.stack([ls(X[:,j]*W[:,None],v[:,j]*W) for j in range(4)]); r=np.linalg.norm(np.einsum('nij,ij->ni',X,c)-v,axis=1); W=1/np.sqrt(1+(r/.12)**2)
 g=np.argsort(r)[:-3]; return np.stack([ls(X[g,j],v[g,j]) for j in range(4)])
def a_inv(x,c):
 b,l,m,n,p=c.T; u=(x-b)/l; k=np.arange(4)
 for _ in range(22):
  z=u[:,J]; r=b+l*u+m*z+n*u*z+p*u**3-x; jac=np.zeros((len(x),4,4)); jac[:,k,k]=l+n*z+3*p*u*u; jac[:,k,J]=m+n*u; u-=np.clip(np.linalg.solve(jac,r[...,None])[...,0],-.65,.65)
 return u,np.max(abs(r))
def b_fit(p):
 u,v=p['u'],p['v']; opts=[]
 funs={'tanh':lambda x,t:x+t[0]*np.tanh(t[1]*x),'rat':lambda x,t:x+t[0]*x/(1+t[1]*x*x),'atan':lambda x,t:x+t[0]*np.arctan(t[1]*x),'sin':lambda x,t:x+t[0]*np.sin(t[1]*x),'soft':lambda x,t:x+t[0]*x/np.sqrt(1+t[1]*x*x),'tanh3':lambda x,t:np.tanh(t[1]*x)+t[0]*x**3,'rat3':lambda x,t:x**3/(1+t[1]*x*x)+t[0]*x,'sinh':lambda x,t:x+t[0]*np.sinh(t[1]*x),'tanh+rat':lambda x,t:x+t[0]*np.tanh(t[1]*x)+t[2]*x/(1+t[3]*x*x)}
 for name,f in funs.items():
  def res(t):
   X=np.c_[ones(u),f(u,t)]; return (X@ls(X,v)-v).ravel()
  best=None
  for q in [.45,.85,1.45,2.1]:
   t0=[-.7,q] if name!='tanh+rat' else [-.6,q,-.15,1.]
   z=least_squares(res,t0,bounds=([-10,.12]+([-10,.12] if len(t0)>2 else []),[10,6]+([10,9] if len(t0)>2 else [])),max_nfev=400)
   e=np.sqrt(np.mean(z.fun**2))
   if best is None or e<best[0]: best=(e,name,z.x,f)
  opts.append(best)
 _,name,t,f=min(opts,key=lambda r:r[0]); return lambda z:np.c_[ones(z),f(z,t)]@ls(np.c_[ones(u),f(u,t)],v)
def c_fit(p):
 u=p['u']; v1=p['v1']; v2=p['v2'];
 def f1(z): return np.c_[ones(z),z*z[:,J]]
 def phi(z,t): return z+t[0]*z*z+t[1]*(1-np.cos(t[2]*z))
 def f2(z,t):return np.c_[ones(z),phi(z,t)]
 def res(t):
  X=f2(u,t); return (X@ls(X,v2)-v2).ravel()
 t=least_squares(res,[.35,.06,2.5],max_nfev=300).x
 c1=ls(f1(u),v1); c2=ls(f2(u,t),v2); return lambda z:f1(z)@c1,lambda z:f2(z,t)@c2

from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


def _features(x, a, b, c, d):
 u, _ = a_inv(x, a)
 p, q, r = b(u), c(u), d(u)
 return [u, p, q, r, c(p), d(p), b(r), c(r), d(q), b(q), b(p), d(r)]


def _kernels(blocks, n):
 gs=[]; ds=[]
 for f in blocks:
  f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7)
  g=(f@f[:n].T)/f.shape[1]
  gs.append(g)
  ss=np.sum(f*f,axis=1)/f.shape[1]
  ds.append(np.maximum(ss[:,None]+ss[None,:n]-2*g,0))
 u,p,q,r,cp,dp,br,cr,dq,bq,bp,dr=gs
 A=[u,p,q,r,cp,dp,br,cr,dq,bq,bp,dr,u**2,u**3,p**2,p**3,u*p,u*u*p,u*p*p,u*q,p*q,u*r,p*r,u*cp,p*cp,u*dp,p*dp,q*r]
 for idx in [(0,1),(0,1,4,5),(0,1,2,3,4,5,6,7,8,9)]:
  dist=np.mean([ds[i] for i in idx],axis=0)
  A.extend(np.exp(-dist/(2*l*l)) for l in [1.,2.,4.])
 return np.array([k/max(1e-9,np.mean(np.diag(k[:n]))) for k in A])


def _readout(blocks, y, traj):
 n=len(y); K=_kernels(blocks,n); eye=np.eye(n)
 Y=np.c_[y,traj]; mu=Y.mean(0); V=Y-mu
 U,S,W=np.linalg.svd(V[:,1:],full_matrices=False)
 Z=np.c_[V[:,0]/max(V[:,0].std(),1e-8),U[:,0]*np.sqrt(n)]
 # Estimate relevance of calibrated feature groups using training labels only.
 Kt=np.ascontiguousarray(K[:,:n]); flat=Kt.reshape(len(K),-1)
 def solve(z, objective=False):
  w=np.exp(z[:-1]); cov=(w@flat).reshape(n,n)+np.exp(z[-1])*eye+1.
  ch=cho_factor(cov,check_finite=False); alpha=cho_solve(ch,Z,check_finite=False)
  if not objective:
   beta=cho_solve(ch,V,check_finite=False)
   return (np.einsum('i,ijk->jk',w,K[:,n:])+1.)@beta+mu
  inv=cho_solve(ch,eye,check_finite=False)
  D=Z.shape[1]*inv-alpha@alpha.T
  val=.5*np.sum(Z*alpha)+Z.shape[1]*np.log(np.diag(ch[0])).sum()
  prior=.035*np.sum((z[:-1]+4.)**2)
  grad=.5*np.r_[(flat@D.ravel())*w,np.trace(D)*np.exp(z[-1])]
  grad[:-1]+=.07*(z[:-1]+4.)
  return val+prior,grad
 z=minimize(lambda z:solve(z,True),np.r_[np.full(len(K),-4.),-6.],jac=True,method='L-BFGS-B',bounds=[(-12,7)]*len(K)+[(-10,0)],options={'maxiter':130}).x
 return solve(z)


def run_node(inputs: dict) -> dict:
 study=inputs.get('__input__',inputs)
 tr=study['train']; te=study['test']; n=len(tr['x'])
 a=a_fit(study['panel_a']); b=b_fit(study['panel_b']); c,d=c_fit(study['panel_c'])
 f=_features(np.vstack((tr['x'],te['x'])),a,b,c,d)
 pred=_readout(f,tr['y'],tr['traj'])
 y, state=pred[:,0],pred[:,1:]; end=state[:,-1]
 return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
