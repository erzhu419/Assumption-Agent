exec(open('/tmp/cv_chamber.py').read().split('for label,DD')[0])
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
for count in (1,2,3):
 for sources in combinations(['train','test','diag'],count):
  label='+'.join(sources)
  ref={k:np.concatenate([dict(train=tr,test=te,diag=g)[s][k] for s in sources]) for k in keys}
  x=g['coupon_features_raw'].copy(); xt,trx=archive._standardize(tr,te)
  for l in np.unique(lab):
   c=ref['ref_coupons'][ref['lab_id']==l]; m=np.nanmedian(np.nanmean(c,0)); s=np.nanmedian(np.nanstd(c,0));
   if s==0: s=np.nanmedian(np.nanstd(joined['ref_coupons'][joined['lab_id']==l],0))
   x[lab==l]=(x[lab==l]-m)/s
  x=np.where(np.isfinite(x),x,Xall); DD=np.c_[np.ones(n),x]@th[2:]
  fix=np.r_[0.,0.,np.log(7.2),.22,.79,0.,0.]
  e=chamber(fix,DD,'unity')-target
  print(label, 'strain/rate RMSE',np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5),'median final/rate',np.median(abs(e),0)[-2:],'per-coupon final error',e[:,-2])
print('diagnostic extreme lab refs/features')
for i in [1,9,10,15]: print(i,g['coupon_features_raw'][i],g['ref_coupons'][i])
