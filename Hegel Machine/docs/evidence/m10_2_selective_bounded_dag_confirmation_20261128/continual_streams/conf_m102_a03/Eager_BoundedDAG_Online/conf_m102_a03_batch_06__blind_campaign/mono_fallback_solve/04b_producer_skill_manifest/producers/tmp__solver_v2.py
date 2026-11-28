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

def _scalar(z,b,p,q,T,y):
 from scipy.linalg import eigh
 from scipy.optimize import minimize
 from sklearn.linear_model import ARDRegression
 n=len(y); N=len(z); H=np.eye(n)-np.ones((n,n))/n; yc=y-y.mean(); candidates=[]
 def pack(a):
  m=a[:n].mean(0); d=a[:n].std(0); keep=d>1e-8; return (a[:,keep]-m[keep])/d[keep]
 def gram(a):
  a=pack(a); return (a@a.T)/max(1,a.shape[1])
 def kernel(K):
  # Profile the intercept; its leverage must be included in PRESS.
  G=H@K[:n,:n]@H; ev,U=eigh((G+G.T)*.5); ev=np.maximum(ev,0); ut=U.T@yc
  alpha=np.geomspace(2e-7,.3,30); fac=1/(ev[:,None]+alpha); a=U@(ut[:,None]*fac); hu=H@U; inv=(hu**2)@fac; e=a/np.maximum(inv,1e-12); pred=y.mean()+(K[n:,:n]-K[:n,:n].mean(0))@a; err=np.mean(abs(e),0)
  for j in np.argsort(err)[:1]: candidates.append((err[j],e[:,j],pred[:,j]))
 def sep(a):return np.c_[a,a*a,a**3,a**4,np.tanh(2*a),np.tanh(4*a),np.sqrt(.12+a*a),np.sin(2.5*a),np.cos(2.5*a)]
 def crosses(a):
  i,j=np.triu_indices(a.shape[1],1); return a[:,i]*a[:,j]
 base=gram(sep(z)); cx=gram(crosses(z)); polyz=gram(poly(z)[:,1:]);
 for a in [z,b,np.c_[z,b],np.c_[z,p],np.c_[z,q],np.c_[z,b,q],np.c_[z,b,p,q],np.c_[z,T],np.c_[z,b,T],np.c_[z,b,p,T],np.c_[z,q,T]]:
  s=gram(sep(a)); poly2=gram(poly(a)[:,1:]); poly3=gram(np.c_[poly(a)[:,1:],crosses(a)*a[:,np.triu_indices(a.shape[1],1)[0]],a**3]);
  for m in [0,.03,.3,3]:
   for K in [s,poly2,poly3,poly2+.4*s,poly3+.4*s]:kernel(K+m*base+.05*cx)
  ds=(a[:,None,:]-a[None,:,:])**2; width=max(np.median(ds.sum(2)),1e-8)
  for k in [.025,.08,.25,.6,1.4,3.]:
   R=ds.sum(2)*k/width; root=np.sqrt(3*R)
   for K in [np.exp(-R),(1+root)*np.exp(-root),np.exp(-ds*k*a.shape[1]/width).mean(2)]:
    for m in [0,.03,.3,3]:kernel(K+m*base+.05*cx)
 for a in [z,b,np.c_[z,b],np.c_[z,q],np.c_[z,b,q,T]]:
  F=pack(np.c_[sep(a),crosses(a),crosses(np.tanh(1.25*a)),crosses(np.sin(2*a))]);
  # Sparse algebraic model; use its fitted precision for a conditional PRESS estimate.
  for cut in [35,300,3000]:
   ar=ARDRegression(threshold_lambda=cut).fit(F[:n],y); keep=ar.lambda_<cut
   if not np.any(keep):continue
   D=F[:,keep]; tr=D[:n]; L=np.diag(ar.lambda_[keep]/ar.alpha_); C=np.linalg.solve(tr.T@tr+L,tr.T); res=(y-y.mean()-tr@(C@yc))/np.maximum(1-1/n-np.sum(tr*C.T,1),.03); pred=y.mean()+D[n:]@(C@yc); candidates.append((np.mean(abs(res))+.003,res,pred))
 # Profile a denominator against compact linear and quadratic numerators.
 for a in [z,b,np.c_[z,b],np.c_[z,T],np.c_[z,b,T]]:
  aa=pack(a); pairs=crosses(aa); nums=[np.c_[aff(aa),pairs],np.c_[aff(aa),pairs,aa**2,aa**3],np.c_[aff(aa),pairs,aa**2,np.tanh(2*aa),np.sqrt(.2+aa*aa)]]
  ds=[aa[:,j] for j in range(aa.shape[1])]; ds+= [aa[:,i]+sg*aa[:,j] for i in range(min(8,aa.shape[1])) for j in range(i+1,min(8,aa.shape[1])) for sg in [-1,1]]
  for d in ds:
   for k in [.1,.35,.9]:
    K=gram(nums[0]/(1+k*d[:,None]**2));kernel(K+.2*base+.05*polyz)
  for F in nums:
   D=pack(F); kernel(gram(D)+.1*base)
 candidates.sort(key=lambda x:x[0]); best=candidates[:65]; chosen=[]
 for ent in best:
  if not any(np.mean(abs(ent[1]-old[1]))<.006 for old in chosen):chosen.append(ent)
  if len(chosen)>=18:break
 E=np.stack([c[1] for c in chosen],axis=1); P=np.stack([c[2] for c in chosen],axis=0); quality=np.array([c[0] for c in chosen]); prior=np.exp(-(quality-quality.min())/.012); prior/=prior.sum();
 def loss(w):return np.mean(np.sqrt((E@w)**2+.009**2))+.003*np.sum((w-prior)**2/np.maximum(prior,.015))
 w=minimize(loss,prior,bounds=[(0,1)]*len(prior),constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP',options={'ftol':1e-9}).x
 return (.78*w+.22*prior)@P

def solve_batch(study):
 s=study; h=s['train']; n=len(h['x']); z=cal(s['panel_a'],np.r_[h['x'],s['test']['x']]); b,p,q=ops(s,z)
 sf=poly(np.c_[z,b])[:,1:]; t=ridge(sf,h['traj'],n,.03)[n:];
 _,_,V=np.linalg.svd(h['traj'],full_matrices=False); T=ridge(sf,h['traj']@V[:1].T,n,.03)
 # For training rows use strictly leave-one-out state features whenever using a learned state.
 for j in range(n):
  order=np.r_[np.arange(j),np.arange(j+1,n),j]; T[j]=ridge(sf[order],(h['traj']@V[:1].T)[order],n-1,.03)[-1]
 y=_scalar(z,b,p,q,T,h['y']); return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
