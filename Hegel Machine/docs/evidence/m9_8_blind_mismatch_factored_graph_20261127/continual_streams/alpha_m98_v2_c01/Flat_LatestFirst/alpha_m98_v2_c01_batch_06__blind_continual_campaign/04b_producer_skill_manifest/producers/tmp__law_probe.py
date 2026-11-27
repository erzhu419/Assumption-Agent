exec(open('/tmp/corrected_probe.py').read().split("print('diagnostic reference")[0])
print('times',t,'dt',dt,'g keys',list(g))
_,xself=m._features(g,g); ds=np.c_[np.ones(len(h)),xself]@th[2:]; print('drifts pooled vs self',np.c_[dr,ds].round(3))
E=-np.expm1(-t[:,None]/tau)
Y=np.c_[y,r]; F=np.vstack([np.c_[E,t],np.r_[1/tau,1]])
A=np.linalg.lstsq(F,Y.T,rcond=None)[0].T
print('free baseline basis errors and coefficients',np.sqrt(np.mean((Y-A@F.T)**2,axis=0)).round(5));print(np.c_[A,dr,ds,h,p].round(3))
from itertools import product
from scipy.optimize import least_squares

def hist(scale,rev=False):
 q=np.zeros((len(h),2)); seq=range(len(dt)-1,-1,-1) if rev else range(len(dt))
 for j in seq:
  a=np.exp(-dt[j]/(tau*scale)); q=q*a+p[:,j,None]*(1-a)
 return q

def fit(F,idx):
 D=F[idx].reshape(-1,F.shape[-1]); return np.linalg.lstsq(D,Y[idx].ravel(),rcond=None)[0]

def rm(e):return np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]

for choice,reverse,drname in product(('gap','split','hist'),(False,True),('linear','exp','rational','log','exp+linear')):
 def basis(pa):
  teff=tau*np.exp(pa[:2]); tf=np.r_[t,0.]; T=-np.expm1(-tf[:,None]/teff);T[-1]=1/teff
  z=hist(np.exp(pa[2]),reverse); ee=[(h[:,None]-z)[:,None,:]*T] if choice=='gap' else [h[:,None,None]*T,z[:,None,:]*T]
  if choice=='hist':ee+=[T[None,:,:]*np.ones((len(h),1,1))]
  k=np.exp(pa[3]); L=tf.copy(); L[-1]=1
  if drname=='linear': V=L
  elif drname in ('exp','exp+linear'): V=-np.expm1(-k*tf)/k;V[-1]=1
  elif drname=='rational': V=tf/(1+k*tf);V[-1]=1
  else: V=np.log1p(k*tf)/k;V[-1]=1
  ee+=[np.c_[ds,np.ones(len(h))][:,None,:]*V[None,:,None]]
  if drname=='exp+linear':ee+=[np.c_[ds,np.ones(len(h))][:,None,:]*L[None,:,None]]
  return np.concatenate(ee,axis=-1)
 allid=np.arange(len(h)); out=[]
 for idx in (allid, *(allid[allid%3!=j] for j in range(3))):
  def fun(pa):
   F=basis(pa); w=fit(F,idx); return (F[idx]@w-Y[idx]).ravel()
  sol=least_squares(fun,[0.,0.,0.,0.],bounds=(-4,4),max_nfev=100,loss='soft_l1',f_scale=.02)
  F=basis(sol.x);w=fit(F,idx);err=F@w-Y
  if len(idx)==len(h): score=rm(err); pars=np.exp(sol.x); coef=w
  else: out.append(err[np.setdiff1d(allid,idx)])
 print(choice,reverse,drname,'fit',score.round(4),'CV',rm(np.vstack(out)).round(4),'params',pars.round(3),'coef',coef.round(3))
