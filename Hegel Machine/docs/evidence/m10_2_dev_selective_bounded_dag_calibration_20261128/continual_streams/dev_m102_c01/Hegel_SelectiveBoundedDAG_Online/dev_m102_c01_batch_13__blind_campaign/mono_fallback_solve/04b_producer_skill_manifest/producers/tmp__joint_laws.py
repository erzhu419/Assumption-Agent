exec(open('/tmp/probe.py').read().split('rec=[]')[0])
from scipy.optimize import least_squares, minimize
from itertools import combinations_with_replacement
from scipy.linalg import solve_triangular

def check(pred, targets=Y):
 return np.array([np.mean(abs(pred(i,o)-targets[o]),axis=0) for i,o in splits])

for name in ('x','z','bz','c2z','c1z'):
 X=charts[name]; lin=one(X); quad=X[:,J]*X[:,K]; F=np.c_[lin,quad];
 for degree in (2,3):
  F=np.column_stack([np.prod(X[:,jj],axis=1) for k in range(degree+1) for jj in combinations_with_replacement(range(4),k)])
  for g in [0.02,.08,.25,.75]:
   for col in [0,1,2]:
    def pred(i,o):
     z=Y[i]; sc=z.std(0); z=z/sc; cols=range(2) if col==2 else [col];
     m=F.shape[1]; D=np.c_[X,quad]; LL=[np.c_[np.tile(np.zeros_like(F[i]),(1,k)),F[i],np.tile(np.zeros_like(F[i]),(1,1-k)),-z[:,k,None]*D[i]] for k in cols]; A=np.concatenate(LL); yy=np.concatenate([z[:,k] for k in cols]); P=np.r_[np.full(2*m,.01 if degree==2 else .04),np.full(D.shape[1],g)]; p0=np.linalg.solve(A.T@A+np.diag(P*P),A.T@yy)[2*m:];
     def loss(p, ret=False):
      h=np.maximum(.18,1+D@p); M=F/h[:,None]; C=np.linalg.solve(M[i].T@M[i]+np.diag(P[:m]**2),M[i].T@z); E=M@C
      if ret:return E[o]*sc
      return np.r_[(E[i][:,cols]-z[:,cols]).ravel(),(P[:m,None]*C[:,cols]).ravel(),g*p, 2*np.minimum(1+D@p-.20,0)]
     q=least_squares(loss,p0,max_nfev=85,ftol=1e-5).x; return loss(q,True)
    E=check(pred); j=0 if col in (0,2) else 1
    if E[0,j]<.036 or E[1:,j].mean()<.029:print('rational',name,degree,g,col,np.round(E.T,5),flush=True)

for name in ['x','z','bz','c1z','c2z']:
 X=charts[name]; D=(X[:,None,:]-X[None,:,:])**2; Z=X@X.T; Q=Z**2; V=1+Z
 for kind in ['rbf','m52','m32','rq','mix']:
  out=[]
  for i,o in splits:
   z=Y[i]; scale=z.std(0); z=z/scale; m=len(i); eye=np.eye(m)
   def kernels(log):
    w=np.exp(log); R=np.einsum('ijk,k->ij',D,w[:4]);
    if kind=='rbf':G=np.exp(-R)
    elif kind=='m52':r=np.sqrt(5*R); G=(1+r+r*r/3)*np.exp(-r)
    elif kind=='m32':r=np.sqrt(3*R); G=(1+r)*np.exp(-r)
    elif kind=='rq':G=(1+R/(2*w[8]))**(-w[8])
    else:G=np.exp(-R)+.5*np.exp(-.2*R)
    return w[4]*V+w[5]*Q+w[6]*G,w[7]
   pp=[]
   for j in range(2):
    def obj(p, ret=False):
     K,r=kernels(p); L=np.linalg.cholesky(K[np.ix_(i,i)]+r*eye); hh=solve_triangular(L,z[:,j],lower=True); alpha=solve_triangular(L.T,hh,lower=False)
     if ret:return (K[np.ix_(o,i)]@alpha)*scale[j]
     return .5*(hh@hh)+np.log(L.diagonal()).sum()+.12*np.sum((p-p0)**2)
    p0=np.log([.6]*4+[.4,.15,1,.001,1]); p=minimize(obj,p0,method='L-BFGS-B',bounds=[(-5,3)]*4+[(-8,5)]*3+[(-12,-3),(-4,5)]).x; pp.append(obj(p,True))
   out.append(np.mean(abs(np.array(pp).T-Y[o]),axis=0))
  print('GP',name,kind,np.round(np.array(out).T,5),flush=True)
