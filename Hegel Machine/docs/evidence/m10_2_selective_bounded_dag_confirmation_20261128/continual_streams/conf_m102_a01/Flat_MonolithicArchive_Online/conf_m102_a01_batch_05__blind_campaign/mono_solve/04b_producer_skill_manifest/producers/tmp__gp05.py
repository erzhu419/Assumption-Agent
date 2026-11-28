import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fitgp(x,Y,n,mode='iso',deg=2,ret=False):
 X=x[:n]; G=x@X.T/4; D=(x[:,None,:]-X[None,:,:])**2; D0=D[:n]; I=np.eye(n); one=np.ones_like(G)
 # Fixed polynomial trend plus stationary residual, with profiled overall variance.
 basis=np.stack((one,G,G**2))[:deg+1]; B=(basis*np.array([1.,1.,.2])[:deg+1,None,None]).sum(0)
 scale=np.sqrt(np.mean(Y**2)); Y=Y/scale
 if Y.ndim==1:Y=Y[:,None]
 m=Y.shape[1]
 def C(p,full=False):
  w=np.exp(p[:2]); a=np.exp(p[2:]); d=D if full else D0; R=w[0]*np.exp(-.5*(d@np.broadcast_to(a,(4,)))); return R,w,a
 def fun(p):
  R,w,a=C(p); K=B[:n]+R+w[1]*I; cf=cho_factor(K); v=cho_solve(cf,Y); ss=np.sum(v*Y)/(n*m); J=(cho_solve(cf,I)-v@v.T/(ss*m))/2; terms=[R,w[1]*I]; terms+= [R*(-.5*(D0@np.broadcast_to(a,(4,))))] if len(a)==1 else [R*(-.5*a[j]*D0[:,:,j]) for j in range(4)]; f=.5*n*np.log(ss)+np.log(np.diag(cf[0])).sum(); jac=np.array([np.sum(J*t) for t in terms]);
  if mode=='ard':
   dp=p[2:]-p[2:].mean(); f+=.6*np.sum(dp**2);jac[2:]+=1.2*dp
  return f,jac
 p0=[-2,-9]+[-1.8]*(4 if mode=='ard' else 1); fit=minimize(fun,p0,jac=True,method='L-BFGS-B',bounds=[(-10,5),(-15,-2)]+[(-5,2)]*(len(p0)-2)); p=fit.x; R,w,a=C(p,True); K=B+R; out=K[n:]@cho_solve(cho_factor(K[:n]+w[1]*I),Y)*scale
 return (out,p) if ret else out
if __name__=='__main__':
 exec(open('/tmp/probe05.py').read().split('for k,w in coords.items():')[0]);
 for key in ['x','h','Bz']:
  W=coords[key]; W=(W-W.mean(0))/W.std(0);
  for mode in ['iso','ard']:
   for degree in [0,1,2]:
    ev=[]
    for k,ix in enumerate(splits):
     i=ix[:52]; j=ix[52:]; a=W[ix]; y0,p=fitgp(a,y[i],52,mode,degree,True); u,sv,v=np.linalg.svd(t[i],full_matrices=False); f,q=fitgp(a,u[:,0]*sv[0],52,mode,degree,True); T=f@v[:1]; e=np.r_[np.abs(y0[:,0]-y[j]).mean(),np.abs(T-t[j]).mean(),np.abs(T[:,-1]-t[j,-1]).mean()];ev.append(e)
     if not k: pars=np.round(p,2),np.round(q,2)
    print(key,mode,degree,'ordered/avg/max',np.round(ev[0],4),np.round(np.mean(ev,axis=0),4),np.round(np.max(ev,axis=0),4),'pars',pars)
