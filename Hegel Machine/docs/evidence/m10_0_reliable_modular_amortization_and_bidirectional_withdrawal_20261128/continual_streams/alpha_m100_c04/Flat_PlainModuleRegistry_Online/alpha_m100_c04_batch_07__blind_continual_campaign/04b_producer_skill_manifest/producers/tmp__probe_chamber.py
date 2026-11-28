import sys, json, pickle, importlib.util
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
s=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/alpha_m100_c04_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
d=calibrate_upstream({**i['dataset'],'chamber_verification_trials':i['campaign_diagnostics']})
tr,te,q=(d[k] for k in ('train','test','chamber_verification_trials'))
dt,t=d['past_dt'],d['eval_times']
x=m._features(tr,te); xd=m._features(tr,q)[len(tr['lab_id']):]
# Apply the train/test reference alignment to the diagnostic panel, and also compare a panel-local alignment.
def diag_x():
    lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); raw=q['coupon_features_raw']; z=np.empty_like(raw)
    L=np.r_[lab,q['lab_id']]; st=np.r_[tr['stratum'],te['stratum'],q['stratum']]
    for j in np.unique(lab):
        r=ref[lab==j]; sel=q['lab_id']==j
        z[sel]=(raw[sel]-np.nanmedian(np.nanmean(r,0)))/np.nanmedian(np.nanstd(r,0))
    _,ix,ct=np.unique(np.column_stack([L,st]),axis=0,return_inverse=True,return_counts=True)
    w=1./ct[ix]; allx=np.vstack([x,z]); mean=np.nansum(w[:,None]*allx,axis=0)/np.sum(w[:,None]*np.isfinite(allx),axis=0)
    return np.where(np.isfinite(z),z,mean)
xq=diag_x()
cal=m._calibration(d['calibration_load_cells']); I=m._inputs(tr,x[:len(tr['lab_id'])],cal); Q=m._inputs(q,xq,cal)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; taus,theta=m._fit(I,dt,t,y,yr)
a,b=m._design(taus,I,dt,t); print('taus,theta',taus,theta, 'train median abs errors',np.median(abs(a@theta-y)),np.median(abs(b@theta-yr)))
a,b=m._design(taus,Q,dt,t); y=q['strain_curves_obs']-q['strain_0'][:,None]; yr=q['instant_rate_obs']; r=y-a@theta; rr=yr-b@theta
print('chamber baseline: per-time median errors',np.median(r,0).round(4),'median abs',np.median(abs(r),0).round(4),'rate median abs',np.median(abs(rr)))
print('diagnostics x differences median/max',np.median(abs(xd-xq),0),np.max(abs(xd-xq),0),'NaNs',np.isnan(q['coupon_features_raw']).sum(0))
past,hold,cov=Q; gap=b[:,:2]*taus; base=np.c_[gap*theta[:2],cov@theta[2:]]; yall=np.c_[y,yr]; w=np.r_[np.ones(len(t)),.65]
def mat(ta):
    return np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]])
def fits(ta,robust=True):
    A=mat(ta)*w[:,None]; Y=yall*w; cs=np.linalg.lstsq(A,Y.T,rcond=None)[0].T
    if robust:
        cs=np.array([least_squares(lambda z:A@z-v,c,loss='soft_l1',f_scale=.025).x for v,c in zip(Y,cs)])
    return cs,(cs@A.T-Y)
cs,er=fits(taus); print('fixed tau unconstrained median abs weighted',np.median(abs(er)),np.median(abs(er),0).round(4));
for st in (taus,taus*np.array([.75,1.3]),taus*np.array([1.3,.75])):
    z=least_squares(lambda z: fits(np.exp(z),False)[1].ravel(),np.log(st),loss='soft_l1',f_scale=.025,bounds=(np.log(.06),np.log(25)),max_nfev=80); tt=np.exp(z.x); print('free taus',tt,'median abs',np.median(abs(fits(tt)[1])))
print('columns: lab stratum hold past[6] x[4] base amplitudes/drift fitted amplitude/drift difference rate-diff'); print(np.c_[q['lab_id'],q['stratum'],hold,past,xq,base,cs-base,rr].round(3))
np.savez('/tmp/chamber_probe.npz',taus=taus,theta=theta,dt=dt,t=t,past=past,hold=hold,cov=cov,y=y,yr=yr,lab=q['lab_id'],stratum=q['stratum'],base=base,cs=cs,xd=xd,raw=q['coupon_features_raw'])
