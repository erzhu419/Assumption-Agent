from m99_probe import *
import json
from scipy.optimize import least_squares
logs=json.load(open(p/'archive_apparatus_log.json')); traces=np.array([v['facility_telemetry_trace'] for v in logs]); print('trace distances:',np.round(np.sqrt(np.mean((traces-traces[-1])**2,axis=(1,2))),3))
# Apply deployment reference statistics to the diagnostic features as a second alignment hypothesis.
refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); labs=np.r_[tr['lab_id'],te['lab_id']]; alt=d['coupon_features_raw'].copy()
for lab in np.unique(labs):
 panel=refs[labs==lab]; i=d['lab_id']==lab
 alt[i]=(alt[i]-np.median(panel.mean(0)))/np.median(panel.std(0))
_,cell,count=np.unique(np.column_stack((labs,np.r_[tr['stratum'],te['stratum']])),axis=0,return_inverse=True,return_counts=True); fill=np.average(xt,axis=0,weights=1/count[cell]); alt=np.where(np.isfinite(alt),alt,fill)
B=hold[:,None]*th[:2]; branch=B@bas.T; rr=r-B@(1/tau); resid=y-branch
print('med residual-time',np.round(np.median(resid,axis=0),4)); print('diag rate and final residual:',np.round(np.column_stack([d['lab_id'],hold,drift,th[2]+alt@th[3:],rr,resid[:,-1]]),4))
for name,u in [('train_diag',drift),('deploy_ref',th[2]+alt@th[3:]),('observed_rate',rr)]:
 for gn,g in [('lin',t),('log',np.log1p(t)),('exp',-np.expm1(-t)),('sqrt',2*(np.sqrt(1+t)-1)),('rational',t/(1+t)),('lin+log',(t+np.log1p(t))/2)]:
  e=resid-u[:,None]*g
  print(name,gn,np.round([np.median(abs(e)),np.quantile(abs(e),.9),np.median(abs(rr-u))],5))
# Estimate time-dependent coefficients independently, with robust loss against curve outliers.
M=np.c_[np.ones(len(hold)),hold,rr,rr**2,hold*rr,mem]
for j in [0,3,7,11]:
 v=resid[:,j]; z=least_squares(lambda a:M@a-v,np.zeros(M.shape[1]),loss='soft_l1',f_scale=.025).x
 print('coefs at',round(t[j],3),np.round(z,4),'median residual',round(np.median(abs(M@z-v)),5))
