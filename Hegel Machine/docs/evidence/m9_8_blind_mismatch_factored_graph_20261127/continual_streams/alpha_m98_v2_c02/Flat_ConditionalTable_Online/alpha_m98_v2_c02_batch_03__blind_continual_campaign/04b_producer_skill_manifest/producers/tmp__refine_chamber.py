import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/chamber_scan.py')
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
print('fit',taus,theta,'diagnostic missing',np.isnan(di['coupon_features_raw']).sum(0),'diagnostic labs',di['lab_id'])
cal=d['calibration_load_cells']; print('cal sizes',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in cal.items()]); err=stress(cal['bridge_raw'])-cal['true_channel_loads'].sum(1);print('cal error rms, max',np.sqrt(np.mean(err**2)),np.max(abs(err)))
# Anchor diagnostic preprocessing to the specified train/test panel.
xx=np.vstack((xtr,xte)); sites=(tr,te); raw=np.vstack([s['coupon_features_raw'] for s in sites]); lab=np.concatenate([s['lab_id'] for s in sites]); ref=np.vstack([s['ref_coupons'] for s in sites]); xdiag=di['coupon_features_raw'].copy()
for j in np.unique(di['lab_id']):
 a=ref[lab==j]; jmask=di['lab_id']==j; xdiag[jmask]=(xdiag[jmask]-np.nanmedian(np.nanmean(a,axis=0)))/np.nanmedian(np.nanstd(a,axis=0))
for j in range(xdiag.shape[1]):
 miss=~np.isfinite(xdiag[:,j]); fill=xx[~np.isfinite(raw[:,j]),j]; xdiag[miss,j]=fill[0] if len(fill) else np.mean(xx[:,j])
for name,x in [('train-test',xdiag),('train-diag',xdi)]:
 drift=theta[2]+x@theta[3:]; mem=(hold[:,None]-z)*theta[:2]; y0=y-mem@decay.T; r0=r-(mem/taus).sum(1)
 print(name,'projected time and rate',np.round(np.r_[drift@y0/(drift@drift),drift@r0/(drift@drift)],5)); print('rate residuals',np.round(r0-drift,5))
 forms={'exp':lambda a: -np.expm1(-a*t)/a,'rat':lambda a:t/(1+a*t),'log':lambda a:np.log1p(a*t)/a,'atan':lambda a:np.arctan(a*t)/a,'tanh':lambda a:np.tanh(a*t)/a,'quad_den':lambda a:t/(1+(a*t)**2)}
 for key, form in forms.items():
  def train(mask,free):
   def fun(p):
    a=p[0]; scale=p[1] if free else 1.; return np.r_[(y0[mask]-drift[mask,None]*scale*form(a)).ravel(),r0[mask]-scale*drift[mask]]
   return least_squares(fun,[1.,1.] if free else [1.],bounds=([.0001,.01],[10,3]) if free else ([.0001],[10]),loss='soft_l1',f_scale=.015).x
  for free in (False,True):
   par=train(np.ones(len(r),bool),free); folds=[]
   for j in np.unique(di['lab_id']):
    te0=di['lab_id']==j; a,*sc=train(~te0,free); scale=sc[0] if free else 1.; folds.append((np.mean(abs(y0[te0]-drift[te0,None]*scale*form(a))),np.mean(abs(r0[te0]-scale*drift[te0]))))
   print(key,free,np.round(par,5),'worst lab',np.round(np.max(folds,axis=0),5))
 print('mem/gap/slopes raw rate projection',np.round(np.linalg.lstsq(np.c_[mem/taus,drift,np.ones(len(drift))],r,rcond=None)[0],5))
