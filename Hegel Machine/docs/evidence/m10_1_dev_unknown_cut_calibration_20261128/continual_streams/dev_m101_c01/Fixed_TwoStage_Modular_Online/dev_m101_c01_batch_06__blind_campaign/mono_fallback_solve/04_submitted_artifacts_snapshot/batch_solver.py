"""Calibrate the inputs and fit separate scalar and time-dependent readouts."""
import numpy as np
from itertools import combinations_with_replacement as cw
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

def poly(x,k):return np.column_stack([np.prod(x[:,j],axis=1) for j in cw(range(x.shape[1]),k)])
def design(x,k):return np.column_stack([poly(x,i) for i in range(k+1)])
def robust(X,y):
 rng=np.random.default_rng(123); m=X.shape[1]; best=None
 for _ in range(900):
  j=rng.choice(len(X),m,replace=False); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]; r=np.linalg.norm(y-X@w,axis=1); i=np.argsort(r); score=r[i[int(.75*len(X))]]
  if best is None or score<best[0]:best=score,i
 j=best[1][:int(.8*len(X))]; w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]
 for _ in range(3):
  r=np.linalg.norm(y-X@w,axis=1); j=r<max(4*np.median(r),.0005); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]
 return w,np.linalg.norm(y-X@w,axis=1)
def predict(x,q,y,groups):
 n=len(x); xs=np.r_[x,q]; D=design(xs,2); D[:,5:]*=.5
 K=[D[:,:1]@D[:,:1].T,D[:,1:5]@D[:,1:5].T,D[:,5:]@D[:,5:].T]
 for a in groups:
  a=np.asarray(a); a=(a-a[:n].mean(0))/np.maximum(.02,a[:n].std(0)); K.append(a@a.T/a.shape[1]);
 K=np.asarray(K); train=K[:,:n,:n]; cross=K[:,n:,:n]; eye=np.eye(n); Y=np.asarray(y); center=Y.mean(0); scale=np.maximum(1e-8,Y.std(0)); Y=(Y-center)/scale
 if Y.ndim==1:Y=Y[:,None]
 def obj(p,ret=False):
  w=np.exp(p); M=np.einsum('i,ijk->jk',w[:-1],train)+w[-1]*eye; fac=cho_factor(M,check_finite=False); c=cho_solve(fac,Y,check_finite=False)
  if ret:return (np.einsum('i,ijk->jk',w[:-1],cross)@c).squeeze()*scale+center
  G=Y.shape[1]*cho_solve(fac,eye,check_finite=False)-c@c.T
  v=.5*np.sum(Y*c)+Y.shape[1]*np.log(fac[0].diagonal()).sum(); g=np.r_[np.einsum('ijk,jk->i',train,G),np.trace(G)]*w*.5
  return v,g
 p=minimize(obj,np.r_[np.full(3,-1.),np.full(len(K)-3,-5.),-8.],jac=True,method='L-BFGS-B',bounds=[(-13,2)]*len(K)+[(-12,-2)],options={'maxiter':100}).x
 return obj(p,True)
def group(q,kind):
 out=[]
 for u in q:
  if kind=='smooth':
   out += [np.c_[np.sin(1.6*u),np.cos(1.6*u)],np.c_[np.sin(2.8*u),np.cos(2.8*u)],poly(u,3),poly(u,4)]
  if kind=='soft':
   out += [u/(1+.83*abs(u)),(u[:,:,None]*(u/(1+.83*abs(u)))[:,None,:]).reshape(len(u),-1),poly(u/(1+.83*abs(u)),2)]
  if kind=='mixed':
   v=np.c_[u/(1+.83*abs(u)),np.cos(1.29*u),np.sin(1.6*u)];out += [v,(u[:,:,None]*v[:,None,:]).reshape(len(u),-1),poly(u,3),poly(u,4)]
 return out
def amplitude(t,Z):
 def fit(r):
  b=-np.expm1(-r*t); c=Z@b/(b@b);return c,b
 def obj(r):
  c,b=fit(r);return np.sum((Z-c[:,None]*b)**2)
 r=minimize_scalar(obj,bounds=(.3,1.5),method='bounded').x
 return r,*fit(r)

def spaces(s,x):
 a=s['panel_a'];A,_=robust(design(a['u'],1),a['v']);A[1:]=np.diag(A[1:].diagonal()); b=s['panel_b'];B=np.linalg.lstsq(design(b['u'],1),b['v'],rcond=None)[0]
 af=lambda X,M:X@M[1:]+M[0];ai=lambda X,M:(X-M[0])@np.linalg.inv(M[1:]);
 return {'x':x,'a':af(x,A),'ai':ai(x,A),'b':af(x,B),'bi':ai(x,B),'ba':af(af(x,A),B),'bai':af(ai(x,A),B),'bia':ai(af(x,A),B),'biai':ai(ai(x,A),B),'ab':af(af(x,B),A),'aib':ai(af(x,B),A),'abi':af(ai(x,B),A),'aibi':ai(ai(x,B),A)}


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, y, z = (train[k] for k in ('x', 'y', 'traj'))
    q = study['test']['x']
    t = study['eval_times']
    sp = spaces(study, np.r_[x, q])
    g1 = group([sp['b']], 'smooth')
    g2 = group([sp['ba']], 'mixed')
    g3 = group([sp['ab']], 'smooth')
    g4 = group([sp['b']], 'mixed')
    yh = (predict(x,q,y,g1) + predict(x,q,y,g2)) / 2
    _, a, b = amplitude(t, z)
    pa = .55*predict(x,q,a,g4) + .45*predict(x,q,a,g3)
    residual = z - a[:, None]*b
    v = np.linalg.svd(residual, full_matrices=False)[2][0]
    pc = predict(x,q,residual@v,g4)
    state = pa[:, None]*b + pc[:, None]*v
    end = state[:, -1]
    return {'y_hat': yh, 'state_pred': state,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}
