from rheology_base import *
import itertools
P,H,X=pq; Y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; n=len(H)
T=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1])); C=np.linalg.lstsq(T,Y.T,rcond=None)[0].T
sets={'tr':tr,'te':te,'q':q}; cat=lambda gs,k:np.concatenate([sets[g][k] for g in gs]); I=np.arange(n); labs=q['lab_id']; missing=np.isnan(q['coupon_features_raw'])
def align(refset,fillset,method='median',idx=I):
 L=cat(refset,'lab_id'); R=cat(refset,'ref_coupons'); Lf=cat(fillset,'lab_id'); S=cat(fillset,'stratum'); F=cat(fillset,'coupon_features_raw').copy(); Q=q['coupon_features_raw'].copy()
 for l in np.unique(L):
  panel=R[L==l]; c=np.median(panel.mean(0)); s=np.median(panel.std(0));
  if method=='flat':c,s=panel.mean(),panel.std()
  if method=='row':c,s=np.median(panel.mean(1)),np.median(panel.std(1))
  F[Lf==l]=(F[Lf==l]-c)/s; Q[labs==l]=(Q[labs==l]-c)/s
 _,g,m=np.unique(np.c_[Lf,S],axis=0,return_inverse=True,return_counts=True); W=1/m[g,None]; fill=np.nansum(F*W,0)/np.sum(np.isfinite(F)*W,0)
 return np.where(missing,fill,Q)
def error(E):return np.array([np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.max(np.sqrt(np.mean(E[:,:-1]**2,axis=1))),np.max(abs(E[:,-1]))])
def cv(F,labels):
 E=np.zeros_like(Y)
 for l in np.unique(labels):
  k=labels!=l; E[~k]=F[~k]@np.linalg.lstsq(F[k],Y[k],rcond=None)[0]-Y[~k]
 return error(E)
from scipy.optimize import least_squares
complete=~missing.any(1); G=Y-H[:,None]*(T[:,:2]@b[:2]); Q=align(('q',),('q',),'median'); D=b[2]+Q@b[3:]
for k in (np.array([.35,1.65,.05]),np.array([.35,1.65,.055]),np.array([.35,1.7,.05])):
 v=T@k; inferred=G@v/(v@v); E=G-D[:,None]*v; print('signature',k,'complete',error(E[complete]).round(5),'implied missing shifts',np.round(np.c_[np.where(missing)[1],(inferred-D)[~complete],(inferred-D)[~complete]/b[3:][np.where(missing)[1]]],4)); print('orthogonal error',error(G-inferred[:,None]*v).round(5))
L=labs
for le in [I,*[I[L!=l] for l in np.unique(L)]]:
 c=np.linalg.lstsq((D[:,None,None]*T)[np.intersect1d(le,I[complete])].reshape(-1,3),G[np.intersect1d(le,I[complete])].ravel(),rcond=None)[0]; print('fit complete signature',c.round(6))
# Distinguish an alternate imputation from a residual constitutive term. The test rows remain unlabeled.
for src in [('q',),('te',),('q','te'),('tr','te')]:
 U=align(('q','te','tr'),src,'median'); print('missing candidates',src,np.round(U[missing],4))
print('missing indices',np.argwhere(missing).tolist()); print('q standardized values used',Q[missing].round(5)); print('key inventory',list(d['test']),list(d['calibration_load_cells']['single_channel_sweeps']))
