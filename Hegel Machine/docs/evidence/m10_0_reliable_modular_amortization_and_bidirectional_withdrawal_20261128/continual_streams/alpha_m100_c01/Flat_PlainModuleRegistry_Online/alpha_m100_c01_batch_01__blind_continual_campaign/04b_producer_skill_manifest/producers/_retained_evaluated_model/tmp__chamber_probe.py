import pickle,numpy as np
import chamber_base as m
np.set_printoptions(precision=5,suppress=True,linewidth=140)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; b=m._bridge(c['bridge_raw'],c['true_channel_loads']); xtr,xte=m._features(tr,te); pp,hh,ff=m._inputs(tr,xtr,b); taus,theta=m._fit(pp,hh,ff,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('fit',taus,theta,'bridge',b)
# Diagnostics use the production reference transform, not a panel refit involving held-out labels.
lab=np.r_[tr['lab_id'],te['lab_id']]; st=np.r_[tr['stratum'],te['stratum']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); x=g['coupon_features_raw'].copy(); xp=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]);
for l in np.unique(lab):
 panel=ref[lab==l]; mu=np.median(panel.mean(0)); sd=np.median(panel.std(0)); xp[lab==l]=(xp[lab==l]-mu)/sd; x[g['lab_id']==l]=(x[g['lab_id']==l]-mu)/sd
_,ic,ct=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1/ct[ic,None]; x=np.where(np.isnan(x),np.nansum(w*xp,0)/np.sum(w*np.isfinite(xp),0),x)
pg,hg,fg=m._inputs(g,x,b); cg,rg=m._design(taus,pg,hg,fg,dt,t); y=g['strain_curves_obs']-g['strain_0'][:,None]; yr=g['instant_rate_obs']; R=y-cg@theta; r=yr-rg@theta
for name,v in [('train',tr['strain_curves_obs']-tr['strain_0'][:,None]-m._design(taus,pp,hh,ff,dt,t)[0]@theta),('diag',R)]: print(name,'RMSE',np.sqrt(np.mean(v*v,0)),'mean',v.mean(0),'sv',np.linalg.svd(v,compute_uv=False))
print('diag rate',np.sqrt(np.mean(r*r)),r.mean(),np.min(r),np.max(r)); print('aug sv',np.linalg.svd(np.c_[R,r],compute_uv=False)); T=np.c_[-np.expm1(-t[:,None]/taus),t]; Tr=np.vstack([T,np.r_[1/taus,1]]); D=np.c_[R,r] @ np.linalg.pinv(Tr.T); print('amps',D); print('temporal residual',np.std(np.c_[R,r]-D@Tr.T,0)); print('corr amp vs [hold,past6,features4,lab,stratum]'); print(np.corrcoef(np.c_[D,hg,pg,x,g['lab_id'],g['stratum']].T)[:3,3:]); print('baseline range',np.min(y),np.max(y)); print('log summaries'); import json
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],a.mean(0),a.std(0))
