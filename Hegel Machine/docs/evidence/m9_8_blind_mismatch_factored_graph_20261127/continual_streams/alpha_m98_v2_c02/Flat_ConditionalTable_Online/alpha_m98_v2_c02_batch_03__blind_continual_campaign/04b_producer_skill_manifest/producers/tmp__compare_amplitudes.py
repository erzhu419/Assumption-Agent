import contextlib, io, runpy, numpy as np
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/refine_chamber.py')
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
for label,site,x,tf in [('train',tr,xtr,t),('chamber',di,xdiag,-np.expm1(-t))]:
 s,rr=m._design(site,np.zeros_like(x),stress,dt,t)(taus)
 err=(site['strain_curves_obs']-site['strain_0'][:,None]-s[:,:,:2]@theta[:2])/tf-(theta[2]+x@theta[3:])[:,None]
 amp=np.median(err,axis=1); rate=site['instant_rate_obs']-rr[:,:2]@theta[:2]-(theta[2]+x@theta[3:]); missing=np.any(np.isnan(site['coupon_features_raw']),axis=1)
 print(label,'lab, n, complete/missing abs amplitude MAE, abs rate-minus-amplitude median')
 for j in np.unique(site['lab_id']):
  mask=site['lab_id']==j
  print(j,mask.sum(),[round(np.mean(abs(amp[q])),5) if q.any() else None for q in (mask&~missing,mask&missing)],round(np.median(abs(rate[mask]-amp[mask])),5))
 if label=='chamber':
  print('diagnostic control mean/std, amp residual by row',np.round(np.c_[site['ref_coupons'].mean(1),site['ref_coupons'].std(1),amp],4))
