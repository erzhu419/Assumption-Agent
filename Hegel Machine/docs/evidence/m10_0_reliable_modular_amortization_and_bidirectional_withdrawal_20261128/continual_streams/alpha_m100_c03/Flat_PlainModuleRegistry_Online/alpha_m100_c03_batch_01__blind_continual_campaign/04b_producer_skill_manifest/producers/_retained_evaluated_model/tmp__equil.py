import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from joint import *
from bench import _features
D=_features(di,di)[:len(h)]; dd=theta[2]+D@theta[3:]; miss=np.isnan(di['coupon_features_raw']).any(1)
print('drift for missing coupons: pooled, diagnostic-imputed',np.c_[dr,dd][miss]);print('diagnostic drift range',dd.min(),dd.max());print('deployment drift range', (theta[2]+X[n:]@theta[3:]).min(),(theta[2]+X[n:]@theta[3:]).max())
for name,c in [('affine',[mat(one,dd)]*3),('linear',[mat(dd)]*3),('quadratic',[mat(one,dd,dd**2)]*3),('hold-scale',[mat(one,dd,h)]*3),('retain-memory',[mat(one,dd,z[:,0]),mat(one,dd,z[:,1]),mat(one,dd)])]:
 print(name)
 for sel in [np.ones(len(h),bool),~miss]:
  pp,w=fitlaw(c,sel); print('fitted',w)
  for group in [di['lab_id'],di['stratum'],miss]:
   print('group folds',[(str(k), *np.round([np.sqrt(np.mean(v[:,:-1]**2)),np.sqrt(np.mean(v[:,-1]**2))],5)) for k in np.unique(group) if np.any(group==k) for v in [(y-fitlaw(c,sel&(group!=k))[0])[group==k]]])
