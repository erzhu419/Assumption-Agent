from graph import *
from scipy.linalg import solve
from scipy.special import erf
from scipy.optimize import least_squares
x=Amap(s['panel_a'])[1](s['train']['x']); raw=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; rng=np.random.default_rng(45); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(4)]]; a,b=np.triu_indices(4); c,d=np.triu_indices(4,1)
def fun(n,u):
 if n==0:return np.tanh(u)
 if n==1:return np.arctan(u)
 if n==2:return u/np.sqrt(1+u*u)
 if n==3:return erf(u)
 if n==4:return np.sin(u)
 if n==5:return u/(1+abs(u))
 if n==6:return np.clip(u,-1,1)
 if n==7:return u/(1+u*u)
def feat(x,n,q,mode):
 z=fun(n,q*x); ZZ=z[:,a]*z[:,b]; XZ=(x[:,:,None]*z[:,None,:]); B=np.c_[np.ones(len(x)),x,z,ZZ];
 if mode==0:return B
 if mode==1:return np.c_[B,XZ.reshape(-1,16)]
 if mode==2:return np.c_[B,x*z]
 if mode==3:return np.c_[B,XZ[:,c,d]+XZ[:,d,c],x*z]
 if mode==4:return np.c_[B,XZ[:,c,d]-XZ[:,d,c],x*z]
 if mode==5:return np.c_[B,x*z,XZ[:,c,d]+XZ[:,d,c], .3*(XZ[:,c,d]-XZ[:,d,c])]
 if mode==6:return np.c_[B,.15*XZ.reshape(-1,16), x*z, (x-z/q)**2, z**3]
 if mode==7:return np.c_[B,x*z,XZ[:,c,d], z**3]
 if mode==8:return np.c_[B,x*z,XZ[:,d,c], z**3]
 if mode==9:return np.c_[B,x*z,(z[:,c]*z[:,d])*(x[:,c]+x[:,d]),z**3]
 if mode==10:return np.c_[B,x*z,(z[:,c]*z[:,d])*(x[:,c]-x[:,d]),z**3]
 if mode==11:return np.c_[B,x*z, (x-z/q)[:,c]*(x-z/q)[:,d], (x-z/q)**2]
 if mode==12:return np.c_[np.ones(len(x)),x,z,ZZ[:,a!=b], x*z,XZ[:,c,d]+XZ[:,d,c]]
 if mode==13:return np.c_[B,x*z, np.sin(z[:,c]+z[:,d]),np.sin(z[:,c]-z[:,d])]
 if mode==14:return np.c_[B,x*z,fun(n,q*(x[:,c]+x[:,d])),fun(n,q*(x[:,c]-x[:,d]))]
R=[]; preds=[]
for n in range(8):
 for q in np.linspace(.7,2.5,19):
  for mode in range(15):
   F=feat(x,n,q,mode)
   for reg in [.0002,.002,.01,.04]:
    for f in [0,1,2,3,4]:
     k=[0,.18,.45,1.,2.][f]; E=[];P=[]
     for i,j in folds:
      Y=np.arcsinh(k*y[i])/k if k else y[i]; W=F[i]; B=solve(W.T@W+reg*np.eye(W.shape[1]),W.T@Y,assume_a='pos'); pred=F[j]@B; pred=np.sinh(np.clip(k*pred,-10,10))/k if k else pred;E.append(np.mean(abs(pred-y[j])));P.append(pred)
     R.append((np.mean(E),E[0],max(E),n,q,mode,reg,k));preds.append(P)
print('Dict AVG',sorted(R)[:10],flush=True); print('Dict WORST',sorted(R,key=lambda r:r[2])[:7],flush=True)
inds=sorted(range(len(R)),key=lambda k:R[k][0]);
for count in [5,15,50,150]:
 idx=inds[:count]; p=np.mean([preds[k] for k in idx],0); print('ensemble',count,[np.mean(abs(w-y[j])) for w,(i,j) in zip(p,folds)],flush=True)
# Search for a rational denominator of a low-dimensional latent coordinate, with variable projection.
RR=[]
for n,q in [(0,1.6),(1,1.55),(2,1.25),(3,1.5)]:
 z=fun(n,q*x); Hs=[z,x,np.c_[z,x]]
 for mode in [0,2,3,5,11]:
  F=feat(x,n,q,mode)
  for h,H in enumerate(Hs):
   for kind in [0,1,2,3]:
    i,j=folds[0]; reg=.001
    def design(p):
     a=H@p[:-1]+p[-1];
     if kind==0:return F/(1+a[:,None]**2)
     if kind==1:return np.c_[F,1/(1+a*a)]
     if kind==2:return np.c_[F,z/(1+a[:,None]**2)]
     return np.c_[F,F[:,5:9]*np.tanh(a[:,None]),np.tanh(a)]
    def fitr(p):
     W=design(p); V=W[i]; m=solve(V.T@V+reg*np.eye(V.shape[1]),V.T@y[i],assume_a='pos'); return W@m,m
    def res(p):
     v,m=fitr(p);return np.r_[v[i]-y[i],np.sqrt(reg)*m,.035*p]
    best=None
    for start in [rng.normal(0,.45,H.shape[1]+1) for _ in range(4)]:
     o=least_squares(res,start,max_nfev=65,bounds=(-4,4)); val=np.linalg.norm(o.fun);
     if best is None or val<best[0]:best=(val,np.mean(abs(fitr(o.x)[0][j]-y[j])),o.x)
    RR.append((best[1],best[0],n,q,mode,h,kind,best[2]))
print('Rat',sorted(RR,key=lambda r:r[0])[:15],flush=True)
# Inspect the panel-B odd scalar map by recovering scaled column directions and residual basis.
u=s['panel_b']['u'];v=s['panel_b']['v']; U=np.c_[np.ones(len(u)),u,u**3]; M=fit(U,v); print('B cubic',M,'resnorm',np.linalg.norm(U@M-v,axis=1)); print('B coefs collinearity',np.sum(M[1:5]*M[5:],axis=1)/np.sum(M[5:]**2,axis=1));
for kind in range(6):
 def ga(q):
  if kind==0:return u**3/(1+q[0]*u*u)
  if kind==1:return np.tanh(q[0]*u**3)
  if kind==2:return u**3/(1+q[0]*abs(u)**q[1])
  if kind==3:return erf(q[0]*u**3)
  if kind==4:return np.tanh(q[0]*u)-q[1]*u
  return u**3/np.sqrt(1+q[0]*u**4)
 for add in [False,True]:
  def r(q):
   Z=np.c_[np.ones(len(u)),u,ga(q)] if add else np.c_[np.ones(len(u)),ga(q)];return (Z@fit(Z,v)-v).ravel()
  o=least_squares(r,[.5,1.5],bounds=([.01,-5],[10,6]),max_nfev=120); print('B alt',kind,add,o.x,np.std(o.fun));
