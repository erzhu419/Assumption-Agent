exec(open('/tmp/chamber_compact.py').read().split("print('dt,time'")[0])
from scipy.optimize import least_squares
X=_features(q,q)[:len(h),1:]; miss=np.isnan(q['coupon_features_raw']); wt=1.; Y=np.c_[y,r];
def basis(u): return np.vstack([np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1]])
F=np.c_[np.ones(len(h)),h,D,p[:,-1]]
def call(a,m):
 u=np.exp(a); B=basis(u); A=np.concatenate([F[:,:,None]*B[None,None,:,k] for k in range(3)],axis=1).transpose(0,2,1); b=np.linalg.lstsq(A[m].reshape(-1,A.shape[-1]),Y[m].ravel(),rcond=None)[0]; return A@b,b
for name,F in [('HD',F[:,:3]),('HDp',F),('HXp',np.c_[F[:,:2],X,F[:,-1]]),('HDmiss',np.c_[F,miss])]:
 for free in (False,True):
  E=np.zeros_like(Y)
  for m in [np.ones(len(h),bool)]+[lab!=l for l in np.unique(lab)]:
   a=np.log(tau)
   if free: a=least_squares(lambda a:(call(a,m)[0][m]-Y[m]).ravel(),a,bounds=(np.log([.2,1.5]),np.log([1.,8.]))).x
   pred,b=call(a,m)
   if m.all():
    print(name,free,'tau',np.exp(a),'fit',np.round(np.sqrt(np.mean((pred-Y)**2,axis=0))[[0,-1]],6),'coef',np.round(b.reshape(3,-1).T,5))
    if name=='HD' and not free: print('rows lab, missing count, residuals first/mid/last/rate',np.round(np.c_[lab,miss.sum(1),(pred-Y)[:,[0,5,-2,-1]]],5))
   E[~m]=pred[~m]-Y[~m]
  print('worst', np.max([[np.mean(E[lab==l,:-1]**2)**.5,np.mean(E[lab==l,-1]**2)**.5] for l in np.unique(lab)],axis=0))
