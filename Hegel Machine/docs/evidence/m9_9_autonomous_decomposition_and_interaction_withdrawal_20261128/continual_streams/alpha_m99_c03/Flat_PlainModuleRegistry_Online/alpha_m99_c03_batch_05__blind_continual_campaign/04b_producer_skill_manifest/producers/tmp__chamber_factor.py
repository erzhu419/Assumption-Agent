from chamber_probe import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
A,G=m._design(tau,inp_d,dt,t)
past,H,F=inp_d; z=H[:,None]-G[:,:2]*tau; r=F@th[2:]; labs=d['lab_id']; missing=np.isnan(d['coupon_features_raw']).any(axis=1)
Y=np.c_[yd,rd]; U=np.vstack(( -np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]
R=Y-H[:,None]*(U@th[:2]); print('after full reset scale',np.linalg.svd(R,compute_uv=False)[:6]); print('labs',np.unique(labs,return_counts=True))
for label,M in [('r',np.c_[F[:,0],r]),('r+z',np.c_[F[:,0],r,z]),('F',F),('F+z+H',np.c_[F,z,H]),('r quadratic',np.c_[F[:,0],r,r*r,z]),('r interaction',np.c_[F[:,0],r,z,z*r[:,None]]),('r and lab',np.c_[F[:,0],r,z,np.eye(labs.max()+1)[labs]])]:
 C=np.linalg.lstsq(M[~missing],R[~missing],rcond=None)[0]; err=R-M@C; print(label,'fit excl missing',score(err[~missing]),'missing',score(err[missing])); print('coeff h=[first,mid,last,rate]',C[:,[0,5,-2,-1]].round(4))
 if label=='r and lab': continue
 e=np.empty_like(R)
 for g in np.unique(labs):
  mask=labs==g; fit=~mask & ~missing; e[mask]=R[mask]-M[mask]@np.linalg.lstsq(M[fit],R[fit],rcond=None)[0]
 print('LOLO complete',score(e[~missing,:-1]),score(e[~missing,-1]),'missing',score(e[missing])); print('worst complete horizon/rate rmse',max(np.sqrt(np.mean(e[~missing,:-1]**2,axis=0))),np.sqrt(np.mean(e[~missing,-1]**2)))
P=np.linalg.lstsq(np.c_[U,T],R.T,rcond=None)[0].T
print('r, amplitudes after reset, features, missing:'); print(np.c_[r,P,F[:,1:],missing].round(4))
print('t',t,'dt',dt)
