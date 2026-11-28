import sys, pickle, json, numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import alpha_m100_c01_batch_01_solver as a
from _verified_upstream import calibrate_upstream
p='/holdout_task/'
i=pickle.load(open(p+'public_batch.pkl','rb'))[0]
d=calibrate_upstream(i['dataset'],json.load(open(p+'_upstream_param_payload.json')))
s=calibrate_upstream(dict(i['dataset'],test=i['campaign_diagnostics']),json.load(open(p+'_upstream_param_payload.json')))['test']
tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; xt,xp=a._features(tr,te); b=np.r_[0.,np.ones(4)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]
tau,th=a._fit(*a._inputs(tr,xt,b),dt,t,y,tr['instant_rate_obs']); print('material',tau,th)
c,r=a._design(tau,*a._inputs(tr,xt,b),dt,t); e=y-c@th; print('train error quantiles',np.quantile(abs(e),[.5,.8,.9,.95,.99,1]),'rate',np.quantile(abs(tr['instant_rate_obs']-r@th),[.5,.9,.99,1]));
# Apply deployment panel normalization estimated jointly from supplied train and test.
lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); x=s['coupon_features_raw'].copy()
for l in np.unique(lab):
 R=refs[lab==l]; m=np.median(R.mean(0)); sd=np.median(R.std(0)); x[s['lab_id']==l]=(x[s['lab_id']==l]-m)/sd; raw[lab==l]=(raw[lab==l]-m)/sd
_,id,n=np.unique(np.c_[lab,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); w=1/n[id,None]; imp=np.nansum(raw*w,axis=0)/np.sum(w*np.isfinite(raw),axis=0); x=np.where(np.isfinite(x),x,imp)
past,hold,f=a._inputs(s,x,b); drift=f@th[2:]; Y=s['strain_curves_obs']-s['strain_0'][:,None]; R=s['instant_rate_obs']; c,r=a._design(tau,past,hold,f,dt,t); base=c@th; rate=r@th; z=hold[:,None]-r[:,:2]*tau; hold_y=-np.expm1(-t[:,None]/tau)@th[:2]; hold_r=sum(th[:2]/tau)
for name,py,pr in [('bench',base,rate),('no-memory linear',hold[:,None]*hold_y+drift[:,None]*t,hold*hold_r+drift),('prior chamber',*a._chamber_response(dict(s,strain_0=np.zeros(len(s['strain_0']))),x,t,b,tau,th))]:
 print(name,'curve',np.quantile(abs(Y-py),[.5,.9,1]),'rate',np.quantile(abs(R-pr),[.5,.9,1])); print('curve residual avg by time',np.round(np.median(Y-py,0),4))
D=np.c_[hold,z,drift]; print('y load/time coeffs',np.round(np.linalg.lstsq(D,Y,rcond=None)[0],4)); print('r coeffs',np.linalg.lstsq(D,R,rcond=None)[0]); print('expected branch coeffs',th[:2]/tau, 'hold',hold_r); print('curve noise after free regression',np.quantile(abs(Y-D@np.linalg.lstsq(D,Y,rcond=None)[0]),[.5,.9,1])); print('rate residual',np.round(R-hold*hold_r-drift,4)); print('note history weight columns',np.round(z[:5],3));
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log keys',list(log) if isinstance(log,dict) else len(log));
if isinstance(log,dict):
 for k,v in log.items():
  print(k, type(v).__name__,len(v) if isinstance(v,(list,dict)) else v)
  if isinstance(v,list) and v and isinstance(v[0],dict):
   for row in v:
    print({j:x for j,x in row.items() if j!='facility_telemetry_trace'}, 'telemetry med',np.median(row['facility_telemetry_trace'],0) if 'facility_telemetry_trace' in row else '')
