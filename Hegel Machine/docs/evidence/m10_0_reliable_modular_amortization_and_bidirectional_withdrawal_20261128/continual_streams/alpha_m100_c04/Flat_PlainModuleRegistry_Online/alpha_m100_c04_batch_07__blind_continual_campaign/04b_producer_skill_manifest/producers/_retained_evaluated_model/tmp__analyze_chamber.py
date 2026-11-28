import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_probe.npz'); globals().update(dict(z)); w=np.r_[np.ones(len(t)),.65]; T=np.r_[t,0]; R=np.r_[np.zeros(len(t)),1]; lin=T+R
rise=lambda u: np.vstack([-np.expm1(-t[:,None]/np.atleast_1d(u)),1/np.atleast_1d(u)])
A=rise(taus); Y=np.c_[y,yr]; D=Y-base@np.c_[A,lin].T; drift=base[:,2]; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; coeff=-np.expm1(-dt[:,None]/taus)*np.exp(-tail[:,None]/taus); state=past@coeff
print('SVD residual',np.linalg.svd(D*w,compute_uv=False).round(4)); print('SVD total',np.linalg.svd(Y*w,compute_uv=False).round(4))
for x in [cov[:,1:],xd]:
 d=theta[2]+x@theta[3:]; B=(hold[:,None]-state)*theta[:2]@A.T+d[:,None]*lin; F=Y-B
 for name,G in [('drift',d[:,None]),('drift+const',np.c_[d,np.ones(len(d))]),('drift+hold+state',np.c_[d,hold,state]),('full linear',np.c_[cov[:,0],x,hold,past])]:
  p=np.linalg.lstsq(G,F,rcond=None)[0]; e=F-G@p; loo=e/(1-np.sum(G*np.linalg.pinv(G).T,axis=1))[:,None]; print('align',x is xd,name,'rmse',np.sqrt(np.mean(e**2,axis=0)).round(4),'loo',np.sqrt(np.mean(loo**2,axis=0)).round(4)); print('coupling',p.round(4) if G.shape[1]<5 else p[:,[0,5,11,12]].round(4))
 print('drift fit through fixed modes',np.linalg.lstsq(d[:,None],cs,rcond=None)[0].round(4))
 U=np.c_[np.ones(len(d)),d,hold,state]; print('free amp fit',np.linalg.lstsq(U,cs,rcond=None)[0].round(4))
