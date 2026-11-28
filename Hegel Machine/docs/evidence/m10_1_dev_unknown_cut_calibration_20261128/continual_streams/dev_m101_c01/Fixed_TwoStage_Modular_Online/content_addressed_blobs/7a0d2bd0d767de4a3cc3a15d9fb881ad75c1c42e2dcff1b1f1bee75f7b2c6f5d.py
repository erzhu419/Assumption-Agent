"""Predict scalar and trajectory outputs from the measured training pairs."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
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

from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb

def poly(x,d,start=0):
 return np.column_stack([np.prod(x[:,j],axis=1) for k in range(start,d+1) for j in comb(range(4),k)])

def rat(x,y,q,kind):
 order,mode,reg=kind; X,T=poly(x,order),poly(q,order); Z,W=poly(x,2,1),poly(q,2,1)
 n=len(x); r2=np.sum(x*x,1); r2q=np.sum(q*q,1)
 def denominator(p,V,r):
  if mode=='rad': return 1+(p[0]+V[:,:4]@p[1:5]+r*p[5])**2
  if mode=='diag': return 1+(p[0]+V[:,:4]@p[1:5]+V[:,[4,8,11,13]]@p[5:])**2
  if mode=='full': return 1+(p[0]+V@p[1:])**2
  if mode=='rad2': return 1+(p[0]+V[:,:4]@p[1:5])**2+np.exp(p[5])*r
  if mode=='rad3': return 1+(p[0]+V[:,:4]@p[1:5])**2+np.exp(p[5])*r*r
  if mode=='qrad': return np.exp(np.r_[p[:4],p[4]]@np.c_[V[:,:4],r].T)
  return 1+np.sum((V[:,:4]@p[:8].reshape(4,2)+p[8:])**2,axis=1)
 m={'rad':6,'diag':9,'full':15,'rad2':6,'rad3':6,'qrad':5,'rank2':10}[mode]
 regX=np.r_[np.full(15,.0001),np.full(X.shape[1]-15,reg)]; regZ=np.r_[np.full(min(5,m),.0005),np.full(max(0,m-5),reg)]
 def residual(p,ret=False):
  P=X/denominator(p,Z,r2)[:,None]; c=np.linalg.lstsq(np.r_[P,np.diag(regX)],np.r_[y,np.zeros(len(regX))],rcond=None)[0]; return c if ret else np.r_[P@c-y,regX*c,regZ*p]
 rng=np.random.default_rng(48); starts=[rng.normal(0,.25,m) for _ in range(5)]; starts.append(np.r_[.45,0,0,.25,.15,.10,np.zeros(max(0,m-6))][:m]);
 if mode in ('rad2','rad3'):
  for p in starts:p[5]=-2
 best=min((least_squares(residual,p,bounds=(-8,8),max_nfev=100,ftol=2e-7) for p in starts),key=lambda f:f.cost)
 return (T@residual(best.x,True))/denominator(best.x,W,r2q), np.mean(abs(best.fun[:n])),best.x


def _components(x, y, z, q):
    scalar = np.column_stack([
        rat(x, y, q, (2, 'rad3', .035))[0],
        rat(x, y, q, (3, 'rad2', .25))[0],
        rat(x, y, q, (3, 'qrad', .25))[0],
        gp(x, y, q, 'rq')[0][:, 0],
    ])
    basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    amplitudes = z @ basis.T
    state = np.stack([
        gp(x, amplitudes, q, 'poly')[0] @ basis,
        gp(x, amplitudes, q, 'rbf')[0] @ basis,
        np.column_stack([gp(x, a, q, 'rbf')[0][:, 0] for a in amplitudes.T]) @ basis,
    ])
    return scalar, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    scalar, state = _components(train['x'], train['y'], train['traj'], study['test']['x'])
    y_hat = scalar @ np.array([.25, .30, .30, .15])
    state_pred = np.average(state, axis=0, weights=[.4, .2, .4])
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
