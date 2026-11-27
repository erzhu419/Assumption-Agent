from m99_probe import *
from collections import defaultdict
print('tau, theta:',tau,beta)
for name,xx in [('joint',X),('train_only',m._features(tr,tr)[0])]:
 tt,bt=m._fit(xx,p,h,dt,t,y,r); aa,bb=m._design(tt,xx,p,h,dt,t)
 print('train',name,'tau',np.round(tt,5),'theta',np.round(bt,5),'curve median absolute',np.median(abs(aa@bt-y)),'rate median absolute',np.median(abs(bb@bt-r)))
raw=c['coupon_features_raw']; missing=~np.isfinite(raw); valid=~missing.any(1); xs=m._features(c,c)[0]; rate0=bc[:,:2]@beta[:2]; residual=(rc-rate0)-(beta[2]+xs@beta[3:]); strata=c['stratum']; labs=c['lab_id']
print('chamber differential missing-feature evidence [column, lab, stratum, implied impute, actual impute]')
for j,k in zip(*np.nonzero(missing)):
 print(k,labs[j],strata[j],*(np.round([xs[j,k]+residual[j]/beta[3+k],xs[j,k]],4)))
print('chamber empirical rates by complete/missing', [np.sqrt(np.mean(residual[u]**2)) for u in (valid,~valid)])
