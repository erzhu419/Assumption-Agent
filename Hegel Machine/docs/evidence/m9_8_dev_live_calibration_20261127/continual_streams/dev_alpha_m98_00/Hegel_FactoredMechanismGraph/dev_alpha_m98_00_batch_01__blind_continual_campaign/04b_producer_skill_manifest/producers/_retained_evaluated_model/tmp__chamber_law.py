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
G=Y-H[:,None]*(T[:,:2]@b[:2]); complete=~missing.any(1)
print('lab panels: rows are group, lab, center, scale, raw missing fraction')
for name,g in sets.items():
 for l in np.unique(g['lab_id']):
  k=g['lab_id']==l; r=g['ref_coupons'][k]; print(name,l,np.round([np.median(r.mean(0)),np.median(r.std(0)),np.isnan(g['coupon_features_raw'][k]).mean()],5))
for r,f in [(('q',),('q',)),(('q',),('te','q')),(('te',),('te',)),(('te','q'),('te','q'))]:
 V=align(r,f,'median'); D=b[2]+V@b[3:]
 for kname,k in [('complete',complete),('all',np.ones(n,bool))]:
  coef=np.linalg.lstsq((D[:,None,None]*T)[k].reshape(-1,3),G[k].ravel(),rcond=None)[0]; E=D[:,None]*(T@coef)-G
  print(r,f,kname,'drift multipliers',coef.round(5),'complete and missing errors',error(E[complete]).round(5),error(E[~complete]).round(5))
 # Implicit shift along the bench drift vector, fitted per coupon using one common temporal signature.
 a0=np.array([.35,1.8,.05]); v=T@a0; inferred=G@v/(v@v); print('inferred D minus candidate',np.round(np.c_[labs,missing.sum(1),inferred-D],4).tolist())
# Jointly estimate stress-proportional response and chamber-specific scalar drift signature on complete coupons.
D=b[2]+align(('q',),('q',),'median')@b[3:]; S=np.stack((H,D),1); A=(S[:,None,:,None]*T[None,:,None,:]).reshape(n,len(t)+1,6)
for kname,k in [('complete',complete),('all',np.ones(n,bool))]:
 c=np.linalg.lstsq(A[k].reshape(-1,6),Y[k].ravel(),rcond=None)[0]; print('two-input law',kname,c.reshape(2,3).round(5),'errors',error(A@c-Y).round(5));
 E=np.zeros_like(Y)
 for l in np.unique(labs):
  m=k&(labs!=l); ch=np.linalg.lstsq(A[m].reshape(-1,6),Y[m].ravel(),rcond=None)[0]; E[labs==l]=A[labs==l]@ch-Y[labs==l]
 print('held-out labs',kname,[(int(l),error(E[labs==l]).round(5).tolist()) for l in np.unique(labs)])
