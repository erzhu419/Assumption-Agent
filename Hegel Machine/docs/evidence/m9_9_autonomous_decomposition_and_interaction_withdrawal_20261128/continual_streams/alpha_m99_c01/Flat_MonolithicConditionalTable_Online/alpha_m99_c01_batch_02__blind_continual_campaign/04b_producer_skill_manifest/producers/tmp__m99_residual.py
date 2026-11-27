from m99_probe import *
from itertools import combinations
xs,_=m._features(c,c)
E=-np.expm1(-t[:,None]/tau); Y=np.c_[yc,rc]; base=np.c_[ac[:,:,:2]@beta[:2],bc[:,:2]@beta[:2]]
T=np.r_[np.c_[E,t],np.r_[1/tau,1.][None,:]]; U=T[:,:2]-T[:,2:3]/tau
labs=c['lab_id']; miss=~np.isfinite(c['coupon_features_raw']); ds=beta[2]+xs@beta[3:]
fit=np.linalg.lstsq((ds[:,None,None]*T).reshape(-1,3),(Y-base).ravel(),rcond=None)[0]; e=base+(ds[:,None]*fit)@T.T-Y
print('fitted chamber coefficients (two transient, one steady)',fit,'initial rate multiplier',fit@T[-1]); print('per coupon: lab, missing count, drift, mean signed strain residual, final residual, rate residual'); print(np.round(np.c_[labs,miss.sum(1),ds,e[:,:-1].mean(1),e[:,-2:]],5))
# Fit on complete coupons first to distinguish fit bias caused by imputation from a constitutive term.
sel=~miss.any(1); fits=[]
for name,u in [('all',np.ones(len(ds),bool)),('complete',sel)]:
 q=(ds[:,None,None]*U).reshape(len(ds),len(t)+1,2); target=Y-base-ds[:,None]*T[:,2]
 b=np.linalg.lstsq(q[u].reshape(-1,2),target[u].ravel(),rcond=None)[0]; err=q@b-target
 print(name,'conserved fractions',b/tau,'steady multiplier',1-(b/tau).sum(),'curve/rate rmse complete',np.sqrt(np.mean(err[sel,:-1]**2)),np.sqrt(np.mean(err[sel,-1]**2)),'missing',np.sqrt(np.mean(err[~sel,:-1]**2)),np.sqrt(np.mean(err[~sel,-1]**2)))
 refs=c['ref_coupons']; print('diagnostic feature centers',np.round(np.mean(xs,0),4))
print('chamber rate residual after single drift adjustment',np.round((rc-base[:,-1])-ds,4))
print('train/test raw fit check: magnitude of test normalization effect',np.quantile(np.abs((m._features(te,te)[0]-Xe)@beta[3:]),[0,.5,.9,1]))
print('archive telemetry summary (mean, last, within-trace std):')
import json
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 z=np.asarray(row['facility_telemetry_trace']); print(row['completed_campaign_short_name'],np.round(np.vstack((z.mean(0),z[-1],z.std(0))),4))
