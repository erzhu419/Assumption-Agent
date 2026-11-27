import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/refine_chamber.py')
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
mem=(hold[:,None]-z)*theta[:2]; y0=y-mem@decay.T; r0=r-(mem/taus).sum(1); f=-np.expm1(-t)
for name,x0 in [('train-test',xdiag),('train-diag',xdi)]:
 dr=theta[2]+x0@theta[3:]; e=y0-dr[:,None]*f; er=r0-dr
 print(name,'cols lab,missing,features,predicted drift,inferred drift,rate residual,early/mid/late residual,off-shape median absolute deviation')
 print(np.round(np.c_[di['lab_id'],np.isnan(di['coupon_features_raw']).sum(1),x0,dr,np.median(y0/f,axis=1),er,e[:,0],e[:,5],e[:,-1],np.median(abs(y0-np.median(y0/f,axis=1)[:,None]*f),axis=1)],4))
 print('curve residual / rate residual correlation by time',np.round(np.corrcoef(e.T,er)[-1,:-1],3))
for key,site in [('tr',tr),('te',te)]:
 a=site['aux_chamber_telemetry']; print(key,'aux percentiles',np.round(np.percentile(a,[0,25,50,75,100],axis=0),3)); print(key,'feature missing rates',np.isnan(site['coupon_features_raw']).mean(0))
print('rate rounding extremes',np.sort(r)[-5:]); print('diagnostic time',t)
