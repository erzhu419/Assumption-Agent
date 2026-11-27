import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m99_rank2 import *
from itertools import combinations
lab=g['lab_id']; missing=np.isnan(g['coupon_features_raw']); clean=~missing.any(1)
b=basis(1.)[:,0]; amp=(E+drift[:,None]*T)@b/(b@b)
print('transient time',u)
blocks={'tr':tr,'g':g,'te':te};
for k in (1,2,3):
 for names in combinations(blocks,k):
  panel=np.vstack([blocks[j]['ref_coupons'] for j in names]); ids=np.concatenate([blocks[j]['lab_id'] for j in names]);
  X=g['coupon_features_raw'].copy(); print('panel',names,end=' ')
  for l in np.unique(lab):
   p=panel[ids==l]; mu=np.median(p.mean(0)); sd=np.median(p.std(0)); X[lab==l]=(X[lab==l]-mu)/sd; print('(%0.4f,%0.4f)'%(mu,sd),end=' ')
  e=amp-(theta[2]+X@theta[3:]); print('complete rms',np.sqrt(np.mean(e[clean]**2)))
print('lab drift scaled fits: lab, intercept, gain, max complete error, raw-feature intercept and gain')
for l in np.unique(lab):
 keep=(lab==l)&clean; z=drift-theta[2]; a=np.c_[np.ones(len(lab)),z]; ql=np.linalg.lstsq(a[keep],amp[keep],rcond=None)[0]; R=np.c_[np.ones(len(lab)),g['coupon_features_raw']@theta[3:]]; q0=np.linalg.lstsq(R[keep],amp[keep],rcond=None)[0]; print(l,ql,np.max(abs((a@ql-amp)[keep])),q0)
