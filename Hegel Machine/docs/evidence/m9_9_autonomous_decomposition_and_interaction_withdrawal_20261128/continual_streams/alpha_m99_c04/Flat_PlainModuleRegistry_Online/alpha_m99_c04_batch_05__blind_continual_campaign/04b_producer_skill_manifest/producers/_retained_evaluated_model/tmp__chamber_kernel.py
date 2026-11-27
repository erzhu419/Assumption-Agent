import numpy as np
from scipy.optimize import least_squares
D=dict(np.load('/tmp/chamber_probe.npz')); globals().update(D)
S=np.c_[hold,past]; U=np.c_[np.ones(len(x)),x]; F=np.c_[S,U]
Y=np.c_[r,y]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]
H=lambda ts: -np.expm1(-dt[:,None]/ts)*np.exp(-age[:,None]/ts)
print('intrinsic',tau,theta,'intrinsic load kernels',np.vstack([np.ones(2),-H(tau)])*theta[:2])
def basis(ts,linear=False):
 return np.column_stack([np.r_[1/s,-np.expm1(-t/s)] for s in ts]+([np.r_[1,t]] if linear else []))
def fit(ts,linear,idx):
 A=basis(ts,linear); M=np.einsum('ij,tk->itjk',F[idx],A).reshape(len(idx)*Y.shape[1],-1); B=np.linalg.lstsq(M,Y[idx].ravel(),rcond=None)[0].reshape(F.shape[1],-1); return F@B@A.T,B
ind=np.arange(len(F)); np.set_printoptions(precision=5,suppress=True)
for linear in (False,True):
 for opt in (False,True):
  def estim(idx):
   ts=tau if not opt else np.exp(least_squares(lambda p:(fit(np.exp(p),linear,idx)[0][idx]-Y[idx]).ravel(),np.log(tau),bounds=(np.log([.2,1.5]),np.log([1.2,9]))).x)
   pred,B=fit(ts,linear,idx); return ts,pred,B
  ts,pred,B=estim(ind); print('linear/free',linear,opt,'tau',ts,'RMSE',np.sqrt(np.mean((pred-Y)**2,axis=0))); print('B rows hold, past chronologically, intercept, X:',B)
  for lab in np.unique(labs):
   i=labs==lab; q=estim(ind[~i])[1][i]-Y[i]; print('lab',lab,'rate/curve rmse',np.sqrt(np.mean(q[:,0]**2)),np.sqrt(np.mean(q[:,1:]**2)))
