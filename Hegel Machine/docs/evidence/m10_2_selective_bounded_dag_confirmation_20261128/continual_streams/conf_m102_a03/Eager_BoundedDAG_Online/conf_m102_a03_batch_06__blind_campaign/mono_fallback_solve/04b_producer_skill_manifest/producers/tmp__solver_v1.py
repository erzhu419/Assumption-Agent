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
 s=study; h=s['train']; n=len(h['x']); z=cal(s['panel_a'],np.r_[h['x'],s['test']['x']]); b,p,q=ops(s,z); A=poly(z)
 def f(w):return np.c_[A,np.tanh(aff(z)@w)]
 def fit(w,ret=False):
  D=f(w); c=ls(D[:n],h['y']); return D@c if ret else np.r_[D[:n]@c-h['y'],.003*w]
 w=least_squares(fit,[0,0,0,0,4],max_nfev=160).x; y=fit(w,True)[n:]; t=ridge(poly(np.c_[z,p])[:,1:],h['traj'],n,.03)[n:];return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
