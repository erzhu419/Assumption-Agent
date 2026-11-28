import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from explore import *
from base import _design
from itertools import combinations_with_replacement
y=np.c_[qc,rc]; bb=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]]); c=lstsq(bb,y.T)[0].T
r=theta[2]+xc@theta[3:]; delta=c.copy(); delta[:,:2]-=hc[:,None]*theta[:2]; u=np.c_[r,hc,pc,xc]; names=['r','h']+['p'+str(i) for i in range(6)]+['x'+str(i) for i in range(4)]
print('tau theta',taus,theta,'dt t',dt,t); print('r and adjusted coefficients');print(np.c_[r,delta]);
for inputs in [(0,), (0,1), (0,1,8,9,10,11), tuple(range(8))]:
 for degree in [1,2,3]:
  powers=[()] + [ix for k in range(1,degree+1) for ix in combinations_with_replacement(inputs,k)]
  if len(powers)>16: continue
  m=np.column_stack([np.prod(u[:,ix],axis=1) for ix in powers]); coef=lstsq(m,delta)[0]; hat=(np.linalg.qr(m)[0]**2).sum(1); pred=(delta-m@coef)/(1-hat[:,None]); e=pred@bb.T;print([names[i] for i in inputs],degree,'LOO coeff',rms(pred,0),'curve rate',rms(e[:,:-1]),rms(e[:,-1]),'coef',coef.round(5).tolist())
for j in range(2):
 a=delta[:,j]; q=lstsq(np.c_[np.ones(len(r)),r],a)[0];print('resid r linear',j,np.corrcoef(np.c_[a-np.c_[np.ones(len(r)),r]@q,u].T)[0,1:].round(3))
