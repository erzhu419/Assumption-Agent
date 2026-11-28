import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import chamber_baseline as b
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,di=[ds[k] for k in ('train','test','chamber_verification_trials')]
dt,t=[ds[k] for k in ('past_dt','eval_times')]; n=len(tr['lab_id']); x=b._features(tr,te)
tau,th=b._fit(*b._loads(tr),x[:n],dt,t,b._observations(tr)); B=b._basis(tau,t)
xx=di['coupon_features_raw'].copy(); refs=np.r_[tr['ref_coupons'],te['ref_coupons']]; labs=np.r_[tr['lab_id'],te['lab_id']]; st=np.r_[tr['stratum'],te['stratum']]
for lab in np.unique(labs):
 m=di['lab_id']==lab; r=refs[labs==lab]; xx[m]=(xx[m]-np.median(r.mean(0)))/np.median(r.std(0))
raw=np.concatenate([tr['coupon_features_raw'],te['coupon_features_raw']]).copy()
for lab in np.unique(labs):
 m=labs==lab; r=refs[m];raw[m]=(raw[m]-np.median(r.mean(0)))/np.median(r.std(0))
_,ix,counts=np.unique(np.c_[labs,st],axis=0,return_inverse=True,return_counts=True); w=1/counts[ix]; mu=np.nansum(raw*w[:,None],0)/(np.isfinite(raw)*w[:,None]).sum(0)
xx=np.where(np.isfinite(xx),xx,mu)
p,h=b._loads(di); g=b._gap(tau,p,h,dt); s=th[2]+xx@th[3:]; y=b._observations(di); c=np.c_[g*th[:2],s]; R=y-c@B.T; dc=np.linalg.lstsq(B,R.T,rcond=None)[0].T
