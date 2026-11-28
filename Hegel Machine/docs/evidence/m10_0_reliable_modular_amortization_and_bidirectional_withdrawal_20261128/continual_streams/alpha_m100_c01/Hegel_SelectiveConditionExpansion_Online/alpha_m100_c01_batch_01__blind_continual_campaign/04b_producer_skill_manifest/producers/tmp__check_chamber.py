from chamber import *
from itertools import product
raw=q['coupon_features_raw']; labs=q['lab_id']; strata=q['stratum']; xx=raw.copy(); clean=np.isfinite(xx).all(1)
for l in np.unique(labs):
 m=labs==l; ref=q['ref_coupons'][m]; xx[m]=(xx[m]-np.median(ref.mean(0)))/np.median(ref.std(0))
_, cell, n=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True)
w=1/n[cell,None]; valid=np.isfinite(xx)
mu=np.nansum(w*xx,0)/np.sum(w*valid,0)
z=np.c_[np.ones(len(xx)),np.where(valid,xx,mu)]@W[2:]
base=(h[:,None]*W[:2])@A[:,:2].T
D=z[:,None,None]*A[None,:,:]
def stats(E):
 return np.round([np.mean(E[:,:-1]**2)**.5,np.mean(E[:,-1]**2)**.5,np.max(np.mean(E[:,:-1]**2,axis=1)**.5),np.max(np.abs(E[:,-1]))],5)
for g in [np.arange(len(labs)),labs,strata]:
 E=np.empty_like(y)
 for val in np.unique(g):
  m=g!=val; k=m&clean
  beta=np.linalg.lstsq(D[k].reshape(-1,3),(y-base)[k].ravel(),rcond=None)[0]; E[~m]=(y-base)[~m]-D[~m]@beta
 print('group count',len(np.unique(g)),'complete',stats(E[clean]),'missing',stats(E[~clean]))
for beta in ([.19552,1.90306,.01715],[.2,1.9,.02]):
 E=y-base-D@beta; print('fixed',beta,'complete',stats(E[clean]),'missing',stats(E[~clean]))
print('diagnostic imputation values',mu,'pooled train/test sample mean',np.vstack(_features(tr,te)).mean(0))
