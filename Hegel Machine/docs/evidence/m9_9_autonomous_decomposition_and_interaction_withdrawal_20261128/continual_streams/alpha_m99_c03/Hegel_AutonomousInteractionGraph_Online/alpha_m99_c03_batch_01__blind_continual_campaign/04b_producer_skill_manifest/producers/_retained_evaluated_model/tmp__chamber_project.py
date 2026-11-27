import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_probe.npz'); p,h,x,dt,t,tau,th,Y,v=[z[k] for k in ('past','hold','features','dt','t','taus','theta','curve','rate')]; r=th[2]+x@th[3:]; o=np.ones(len(h)); after=np.cumsum(dt[::-1])[::-1]-dt
m=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-after[:,None]/tau)); y=np.c_[Y,v]
def fit(A, mask, mode):
 def form(par):
  a=np.exp(par) if mode==2 else np.r_[tau[0],np.exp(par)] if mode==1 else tau
  return a,np.vstack([np.c_[-np.expm1(-t[:,None]/a),t],np.r_[1/a,1.]])
 def solve(par):
  a,F=form(par); L=np.einsum('ni,kj->nkij',A[mask],F).reshape(-1,A.shape[1]*3); co=np.linalg.lstsq(L,y[mask].ravel(),rcond=None)[0].reshape(-1,3); return a,co,A@co@F.T
 par=np.log(tau[-mode:]) if mode else None
 if mode: par=least_squares(lambda b:(solve(b)[2][mask]-y[mask]).ravel(),par,bounds=(np.log([.25,1.2][-mode:]),np.log([1.15,12.][-mode:]))).x
 return solve(par)
for tag,A in [('hold,r',np.c_[h,r]),('1,hold,r',np.c_[o,h,r]),('1,hold,r,m',np.c_[o,h,r,m]),('1,hold,x',np.c_[o,h,x]),('1,hold,x,m',np.c_[o,h,x,m])]:
 for mode in range(3):
  a,co,pred=fit(A,np.ones(len(h),bool),mode); E=np.stack([y[i]-fit(A,np.arange(len(h))!=i,mode)[2][i] for i in range(len(h))]); print(tag,mode,np.round(a,5),np.round([np.std(E[:,:-1]),abs(E[:,:-1]).max(),np.std(E[:,-1]),abs(E[:,-1]).max()],5));
  if mode==2: print('coef',np.round(co,5),'rate coef',np.round(co@np.r_[1/a,1],5))
