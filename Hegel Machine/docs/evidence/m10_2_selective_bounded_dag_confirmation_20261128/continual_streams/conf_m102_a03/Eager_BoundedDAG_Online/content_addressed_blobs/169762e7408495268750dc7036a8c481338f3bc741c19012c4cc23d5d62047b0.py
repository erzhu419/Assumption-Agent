"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from scipy.optimize import minimize_scalar, least_squares

def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-11)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x):
 i,j=np.triu_indices(x.shape[1]); return np.c_[aff(x),x[:,i]*x[:,j]]
def cal(p,x):
 u,v=p['u'],p['v']; n=np.roll(u,-1,1); A=np.stack((np.ones_like(u),u,n,u*n),2); w=np.ones(len(u))
 for _ in range(30):
  c=np.stack([ls(A[:,j]*w[:,None],v[:,j]*w) for j in range(4)],1); r=np.linalg.norm(np.einsum('ijk,kj->ij',A,c)-v,axis=1);w=np.sqrt(np.minimum(1,1.3*np.median(r)/np.maximum(r,1e-12)))
 good=r<4*np.median(r)
 def coef(k,ret=False):
  D=np.concatenate((A,np.sin(k*u)[:,:,None]),axis=2)[good]; C=np.stack([ls(D[:,j],v[good,j]) for j in range(4)],1)
  if ret:return C
  return np.mean([np.mean(((D[:,j]@C[:,j]-v[good,j])/(1-(np.linalg.qr(D[:,j])[0]**2).sum(1)))**2) for j in range(4)])
 k=minimize_scalar(coef,bounds=(.12,1.6),method='bounded').x; c=coef(k,True)
 d=np.eye(4); e=np.roll(d,1,axis=1); z=np.linalg.solve(d*(c[1]+k*c[4])[:,None]+e*c[2,:,None],(x-c[0]).T).T
 for _ in range(30):
  n=np.roll(z,-1,1); pred=c[0]+c[1]*z+c[2]*n+c[3]*z*n+c[4]*np.sin(k*z)
  jac=d*(c[1]+c[3]*n+c[4]*k*np.cos(k*z))[:,:,None]+e*(c[2]+c[3]*z)[:,:,None]; z+=np.clip(np.linalg.solve(jac,(x-pred)[...,None])[...,0],-.4,.4)
 return z

def ops(s,z):
 out=[]
 for pn,vn,fn,bounds in [('panel_b','v',np.sin,(2.2,3.3)),('panel_c','v1',np.arcsinh,(1.5,8)),('panel_c','v2',lambda x:np.sqrt(1+x*x),(.5,1.6))]:
  u,v=s[pn]['u'],s[pn][vn]
  def design(u,k):return np.c_[aff(u),fn(k*u)]
  def loss(k):
   Q=np.linalg.qr(design(u,k))[0]; return np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1))[:,None])**2)
  k=minimize_scalar(loss,bounds=bounds,method='bounded').x;out.append(design(z,k)@ls(design(u,k),v))
 return out

def ridge(a,y,n,alpha):
 m=a[:n].mean(0); d=np.maximum(a[:n].std(0),1e-10); a=(a-m)/d; h=a[:n]; b=y[:n]-y[:n].mean(0); return y[:n].mean(0)+a@(h.T@np.linalg.solve(h@h.T+alpha*np.eye(n),b))

def solve_batch(study):
 s=study; h=s['train']; n=len(h['x']); z=cal(s['panel_a'],np.r_[h['x'],s['test']['x']]); b,p,q=ops(s,z)
 sf=poly(np.c_[z,b])[:,1:]; t=ridge(sf,h['traj'],n,.03)[n:];
 _,_,V=np.linalg.svd(h['traj'],full_matrices=False); T=ridge(sf,h['traj']@V[:1].T,n,.03)
 # For training rows use strictly leave-one-out state features whenever using a learned state.
 for j in range(n):
  order=np.r_[np.arange(j),np.arange(j+1,n),j]; T[j]=ridge(sf[order],(h['traj']@V[:1].T)[order],n-1,.03)[-1]
 y=_scalar(z,b,p,q,T,h['y']); return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}

def _scalar(z,b,p,q,T,y):
 """Fit additive and interacting covariance components by profiled likelihood."""
 from scipy.linalg import cho_factor, cho_solve
 from scipy.optimize import minimize
 n=len(y); y=np.asarray(y); ones=np.ones(n); eye=np.eye(n); delta=np.subtract.outer
 def scale(a):
  a=np.asarray(a).reshape(len(z),-1); return (a-a[:n].mean(0))/np.maximum(a[:n].std(0),1e-8)
 def g(a):
  a=scale(a); return a@a.T/a.shape[1]
 def d(a):
  a=scale(a); return np.mean((a[:,None,:]-a[None,:,:])**2,axis=2)
 def rdist(D,w): return np.exp(-D*w)
 def tangent(x):return np.c_[x,x*x,x**3,np.tanh(2.5*x),np.sqrt(.12+x*x),np.sin(2.5*x),np.cos(2.5*x)]
 i,j=np.triu_indices(4,1)
 additive=[g(tangent(z[:,k:k+1])) for k in range(4)]
 full=[]
 for jset in (b,p,q):
  full.append([g(tangent(jset[:,k:k+1])) for k in range(4)])
 zcross=[g(np.c_[z[:,k]*z[:,l],z[:,k]**2*z[:,l],z[:,k]*z[:,l]**2]) for k,l in zip(i,j)]
 zd=d(z); bd=d(b); pd=d(p); qd=d(q); td=d(T); zbd=d(np.c_[z,b]); base=g(tangent(z));
 candidates=[]
 def prep(kernels):
  return np.asarray(kernels).copy()
 def optimize(kernels,prior,noise=-7.,starts=2):
  Ks=prep(kernels); cross=Ks[:,n:,:n]; K=Ks[:,:n,:n].copy(); fl=K.reshape(len(K),-1); tr=fl.shape[1]; prior=np.asarray(prior); basepar=np.r_[prior,noise];
  def obj(a,ret=False):
   w=np.exp(a); A=(w[:-1]@fl).reshape(n,n)+w[-1]*eye
   R=cho_factor(A,check_finite=False); M=cho_solve(R,eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; den=np.maximum(np.diag(G),1e-10); e=al/den; err=np.mean(np.abs(e));
   if ret:return err,e,mean+(w[:-1]@cross.reshape(len(K),-1)).reshape(len(z)-n,n)@al
   ll=.5*(np.sum(np.log(np.diag(R[0])))*2+np.log(1/v)+y@al)
   F=G-np.outer(al,al); grad=.5*w*np.r_[fl@F.ravel(),np.trace(F)]; diff=a-basepar
   return ll+.045*(diff@diff),grad+.09*diff
  for off in [0,-1.5][:starts]:
   init=basepar.copy(); init[:-1]+=off
   opt=minimize(obj,init,jac=True,method='L-BFGS-B',bounds=[(-16,5)]*len(init),options={'maxiter':175,'ftol':1e-9});
   candidates.append(obj(opt.x,True))
 for X,D,aux in [(z,zd,None),(np.c_[z,b],zbd,0),(np.c_[z,p],(zd+pd)/2,1),(np.c_[z,q],(zd+qd)/2,2)]:
  P=poly(X); C=[base,g(P[:,1:]),g(tangent(X))]+[rdist(D,k) for k in [.025,.12,.4,1.3,4]]
  if aux is not None:C+=full[aux]
  optimize(C,[-2,-2,-3,-2,-2,-3,-5,-6]+([-4]*4 if aux is not None else []))
 C=additive+zcross+[g(tangent(b)),g(tangent(p)),g(tangent(q))]+[rdist(D,k) for D in [zd,zbd,(zd+pd)/2,(zd+qd)/2,(zd+td)/2] for k in [.08,.45,1.7]]
 optimize(C,[-2]*4+[-3]*6+[-4]*3+[-3,-4,-6]*5)
 C=additive+full[0]+full[1]+full[2]+zcross+[g(T),g(tangent(T)),g(T*z),g(T*b)]+[rdist(D,k) for D in [zbd,(zd+td)/2,(bd+td)/2] for k in [.08,.5,2]]
 optimize(C,[-2]*4+[-4]*12+[-3]*6+[-3]*4+[-4,-4,-6]*3)
 # Let the data select coordinate scales in a smooth nonseparable covariance.
 for X in [z,np.c_[z,b],np.c_[z,q],np.c_[z,b,T]]:
  X=scale(X); parts=(X[:,None,:]-X[None,:,:])**2; baseK=base+.15*g(poly(z)[:,1:]); terms=parts[:n,:n].reshape(n*n,-1).copy(); test=parts[n:,:n].reshape(-1,X.shape[1]); m=X.shape[1];
  def obj(a,ret=False):
   w=np.exp(a); raw=w[m]*np.exp(-(terms@w[:m])).reshape(n,n); A=raw+w[m+1]*baseK[:n,:n]+w[-1]*eye
   R=cho_factor(A,check_finite=False); M=cho_solve(R,eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; e=al/np.maximum(np.diag(G),1e-10)
   if ret:
    K=w[m]*np.exp(-(test@w[:m])).reshape(len(z)-n,n)+w[m+1]*baseK[n:,:n]; return np.mean(abs(e)),e,mean+K@al
   F=.5*(G-np.outer(al,al)); rr=(F*raw); grad=np.r_[-w[:m]*(terms.T@rr.ravel()),rr.sum(),w[m+1]*np.sum(F*baseK[:n,:n]),w[-1]*np.trace(F)]; diff=a-center
   return .5*(2*np.log(np.diag(R[0])).sum()+np.log(1/v)+y@al)+.065*(diff@diff),grad+.13*diff
  for k in [.12,.5,1.5]:
   center=np.r_[np.log(np.repeat(.25/m,m)),0.,-2.,-7.]; init=center.copy(); init[:m]=np.log(k/m)
   opt=minimize(obj,init,jac=True,method='L-BFGS-B',bounds=[(-12,3)]*m+[(-12,5),(-13,4),(-16,0)],options={'maxiter':180,'ftol':1e-10}); candidates.append(obj(opt.x,True))
 # Fixed broad smooth priors provide a guard against unstable likelihood optima.
 for D in [zd,zbd,(zd+pd)/2,(zd+qd)/2,(zd+td)/2]:
  for k in [.05,.15,.45,1.3]:
   for weight in [.03,.3,3.]:
    A=np.exp(-D*k)+weight*base+.05*g(z[:,i]*z[:,j]);
    for noise in [2e-6,2e-5,2e-4,.002,.02]:
     M=cho_solve(cho_factor(A[:n,:n]+noise*eye,check_finite=False),eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; e=al/np.maximum(np.diag(G),1e-10); candidates.append((np.mean(abs(e))+.002,e,mean+A[n:,:n]@al))
 chosen=[]
 for c in sorted(candidates,key=lambda c:c[0]):
  if not chosen or all(np.mean(abs(c[1]-a[1]))>.007 for a in chosen):chosen.append(c)
  if len(chosen)>=22:break
 E=np.array([c[1] for c in chosen]).T; P=np.array([c[2] for c in chosen]); qual=np.array([c[0] for c in chosen]); w0=np.exp(-(qual-qual.min())/.012); w0/=w0.sum()
 def loss(w):return np.mean(np.hypot(E@w,.012))+.006*np.sum((w-w0)**2/np.maximum(w0,.02))
 out=minimize(loss,w0,method='SLSQP',bounds=[(0,1)]*len(w0),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-9}); w=out.x if out.success else w0
 return (.8*w+.2*w0)@P
