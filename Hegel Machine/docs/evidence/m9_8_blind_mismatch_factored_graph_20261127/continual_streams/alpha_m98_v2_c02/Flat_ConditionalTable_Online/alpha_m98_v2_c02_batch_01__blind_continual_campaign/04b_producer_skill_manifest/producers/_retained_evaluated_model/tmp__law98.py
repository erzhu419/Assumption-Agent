exec(open('/tmp/probe98.py').read().split('for name,X in')[0])
from scipy.linalg import lstsq
q=f@b[2:]; base=(h[:,None]-s@W)*b[:2]; T=basis(u)[:2]
def expb(v):return np.r_[-np.expm1(-t/v),.7/v]
def rm(E):return np.sqrt(np.mean(E**2,axis=0))[[0,5,11,12]]
# Fit stress coefficients and a saturating drift response directly in observation space.
def fit(idx,mode):
 def parts(p):
  if mode==0:return [np.c_[h,s],np.c_[h,s]],T, q[:,None]*expb(p[0])*p[1]
  if mode==1:return [np.c_[h,s],np.c_[h,s]],T,q[:,None]*T[1]*p[0]
  if mode==2:return [np.c_[s],np.c_[s]],T,q[:,None]*expb(p[0])*p[1]
  if mode==3:return [np.c_[h,s],np.c_[h,s]],np.array([expb(p[2]),expb(p[3])]),q[:,None]*expb(p[0])*p[1]
  if mode==4:return [np.c_[h,s],np.c_[h,s]],T,np.zeros_like(Y)
  return [np.c_[h,s],np.c_[h,s,q]],T,np.zeros_like(Y)
 def pred(p,ret=False):
  X,H,Z=parts(p); A=np.concatenate([x[:,None,:]*v[None,:,None] for x,v in zip(X,H)],axis=2); E=Y-Z
  coef=lstsq(A[idx].reshape(-1,A.shape[2]),E[idx].ravel())[0]; P=A@coef+Z
  return (P,coef) if ret else (P[idx]-Y[idx]).ravel()
 seeds=([3.9,2.5],[2.5],[3.9,2.5],[3.9,2.5,*u],[],[])[mode]; bounds=([.15,0,.2,1.5],[20,15,1.5,18]); p=least_squares(pred,seeds,bounds=(bounds[0][:len(seeds)],bounds[1][:len(seeds)]),max_nfev=150).x if seeds else []; P,C=pred(p,True);return P,C,p
for mode in range(6):
 P,C,p=fit(np.arange(len(h)),mode); E=np.stack([fit(np.flatnonzero(np.arange(len(h))!=i),mode)[0][i]-Y[i] for i in range(len(h))]); print('mode',mode,'params',p,'coeff',C,'fit',rm(P-Y),'loo',rm(E))
print('canonical stress kernel',W*b[:2]); print('q vs diag drift',np.c_[q,B[:,2],B[:,1]-base[:,1]]); print('telemetry train/test correlation and range');
for k in ['train','test']:
 z=d[k]['aux_chamber_telemetry'];print(k,np.mean(z,axis=0),np.std(z,axis=0),np.corrcoef(z.T))
