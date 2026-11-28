import numpy as np
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

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y=d['x'],d['y']; n=len(x); folds=[np.arange(56,72),*np.array_split(np.random.default_rng(22).permutation(n),5),np.argsort(np.linalg.norm(x,axis=1))[-14:]]
 for mode in ['rad','diag','full','rad2','rad3','rank2','qrad']:
  for degree,reg in [(2,.035),(3,.08),(3,.25)]:
   scores=[]; trs=[]
   for j in folds:
    i=np.setdiff1d(np.arange(n),j); p,tr,_=rat(x[i],y[i],x[j],(degree,mode,reg));scores.append(np.mean(abs(p-y[j])));trs.append(tr)
   print(mode,degree,reg,np.round(scores,4),round(np.mean(trs),5),flush=True)
