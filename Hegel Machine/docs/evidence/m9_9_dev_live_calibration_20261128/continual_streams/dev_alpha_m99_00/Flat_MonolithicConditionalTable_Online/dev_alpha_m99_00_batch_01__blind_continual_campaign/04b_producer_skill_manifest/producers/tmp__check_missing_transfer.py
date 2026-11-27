import contextlib,io,runpy,numpy as np
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','tr','te','di','t','dt','tau','theta','bridge','v','vr','y','yr','a')})
for split,coupon,obs,rate in [(tr,a,y,yr)]:
 C,R=m._design(tau,coupon,dt,t); miss=np.isnan(split['coupon_features_raw']).any(1); e=obs-C@theta; er=rate-R@theta
 for flag in (False,True):
  q=miss==flag; print('train missing/count',flag,q.sum(),'median late/rate',np.median(e[q,-1]),np.median(er[q]),'median abs',np.median(np.abs(e[q,-1])),np.median(np.abs(er[q])))
x=m._features(tr,te)[len(tr['lab_id']):]; mask=np.isnan(te['coupon_features_raw']).any(1); tmax=t.max(); late=.060623*(-np.expm1(-tmax/.322898))+.962508*(-np.expm1(-tmax/1.496942))+.16156*tmax; rate=.060623/.322898+.962508/1.496942+.16156
print('deployment missing count:',int(mask.sum()),'of',len(mask),'indicator shift at final time and initial rate:',(-.017009*late,-.017009*rate),'pooled rate MAE distance:',.017009*rate*mask.mean())
