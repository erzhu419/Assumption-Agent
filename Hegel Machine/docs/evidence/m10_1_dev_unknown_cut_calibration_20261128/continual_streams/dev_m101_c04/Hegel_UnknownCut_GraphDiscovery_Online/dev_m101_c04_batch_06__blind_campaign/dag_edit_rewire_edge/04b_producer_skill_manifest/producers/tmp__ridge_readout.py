from dag_probe import *
from scipy.optimize import least_squares
from scipy.linalg import qr
from itertools import product
rng=np.random.default_rng(442); X=np.c_[u,z]; Q=basis(u,2); N=52
folds=[(np.setdiff1d(np.arange(72),v),v) for v in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)]]
fs={'sin':np.sin,'tanh':np.tanh,'rat':lambda a:a/(1+a*a),'gaussian':lambda a:np.exp(-a*a),'cos':np.cos}
for mode in ['linear','bilinear','rank1','quadratic']:
 for fn,f in fs.items():
  for count in [1,2]:
   err=[]; tr=[]; fits=[]
   for t,v in folds:
    A,B=Q[t],Q[v]; U=np.linalg.qr(A)[0]; r=Y[t]-U@(U.T@Y[t]);
    if mode=='linear': Z=np.c_[np.ones(len(u)),u]; k=5
    elif mode=='bilinear': Z=np.c_[np.ones(len(u)),u*z]; k=5
    elif mode=='rank1': Z=basis(u,1); k=10
    else: Z=Q; k=15
    def features(p,rows):
     p=p.reshape(k,count); a=Z[rows]@p[:Z.shape[1]]
     if mode=='rank1': a*=Z[rows]@p[5:]
     return f(a)
    def calc(p):
     H=features(p,t); H-=U@(U.T@H); w=np.linalg.lstsq(H,r,rcond=1e-7)[0]; return H@w-r,w
    best=None
    for rep in range(16 if count==1 else 8):
     p0=rng.normal(0,.8,(k,count));
     opt=least_squares(lambda p:calc(p)[0],p0.ravel(),max_nfev=130,ftol=1e-6)
     e,w=calc(opt.x)
     if best is None or np.linalg.norm(e)<best[0]: best=(np.linalg.norm(e),opt.x,w,np.mean(abs(e)))
    e,p,w,mae=best; coef=np.linalg.lstsq(A,Y[t]-features(p,t)@w,rcond=None)[0]; err.append(np.mean(abs(B@coef+features(p,v)@w-Y[v]))); tr.append(mae); fits.append((p,w))
   print(mode,fn,count,np.round(err,5),np.round(tr,5), 'params',np.round(fits[0][0],3),flush=True)
