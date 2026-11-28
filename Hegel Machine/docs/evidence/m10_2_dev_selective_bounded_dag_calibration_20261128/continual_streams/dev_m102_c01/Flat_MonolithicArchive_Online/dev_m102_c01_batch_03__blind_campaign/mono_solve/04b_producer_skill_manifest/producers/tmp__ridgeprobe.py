import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from scipy.optimize import least_squares
from scipy.linalg import lstsq

def basis(q,mode):
 if mode=='tanh':return np.tanh(q)
 if mode=='sin':return np.sin(q)
 if mode=='rat':return q/(1+q*q)
 if mode=='even':return q*q/(1+q*q)
 if mode=='soft':return q/np.sqrt(1+q*q)
 if mode=='exp':return np.exp(-q*q)
 if mode=='cos':return np.cos(q)
 if mode=='mix':return np.c_[np.tanh(q), q*q/(1+q*q)]

def fit(q, y, test, mode, m, degree=1, starts=8, seed=91):
 def base(q):
  if not degree:return np.ones((len(q),1))
  i,j=np.triu_indices(q.shape[1]); return np.c_[np.ones(len(q)),q] if degree==1 else np.c_[np.ones(len(q)),q,q[:,i]*q[:,j]]
 A=base(q); At=base(test); X=np.c_[np.ones(len(q)),q]; Xt=np.c_[np.ones(len(test)),test]; rng=np.random.default_rng(seed); best=None
 y=y.reshape(len(q),-1); scale=np.maximum(y.std(0),1e-5); yy=y/scale
 def feats(p):return np.c_[A,basis(X@p.reshape(X.shape[1],m),mode)]
 def lin(F):return lstsq(F,yy,cond=1e-8,lapack_driver='gelsy')[0]
 def fun(p):
  F=feats(p); w=lin(F); return np.r_[(F@w-yy).ravel(), .001*p]
 for i in range(starts):
  P=rng.normal(0,.45+(.45 if i%2 else 0),(X.shape[1],m)); P[0]*=.5
  opt=least_squares(fun,P.ravel(),max_nfev=120,ftol=1e-7)
  if best is None or opt.cost<best.cost:best=opt
 W=best.x.reshape(X.shape[1],m); F=feats(best.x); beta=lin(F); return (F@beta)*scale,(np.c_[At,basis(Xt@W,mode)]@beta)*scale,W,beta
if __name__=='__main__':
 for k in ('v1','v2'):
  u=s['panel_c']['u']; v=s['panel_c'][k]
  if k=='v2':continue
  for mode in ('tanh','sin','rat','soft','mix'):
   for m in (4,):
    h,q,P,W=fit(u,v,u,mode,m,0,8); print(k,mode,m,np.sqrt(np.mean((v-h)**2)),P.T,flush=True)
 for tag,yy in [('y',y),('y_resid',y- .6*T[:,0]),('traj',T)]:
  for mode in ('tanh','sin','rat','even','soft','exp','mix'):
   for m in (1,2,3,4):
    yh,h,P,W=fit(z[:52],yy[:52],z[52:],mode,m,1,7); h=h.reshape(yy[52:].shape); yh=yh.reshape(yy[:52].shape)
    print(tag,mode,m,np.mean(abs(yh-yy[:52]),axis=0),np.mean(abs(h-yy[52:]),axis=0),np.round(P.T,3),flush=True)
