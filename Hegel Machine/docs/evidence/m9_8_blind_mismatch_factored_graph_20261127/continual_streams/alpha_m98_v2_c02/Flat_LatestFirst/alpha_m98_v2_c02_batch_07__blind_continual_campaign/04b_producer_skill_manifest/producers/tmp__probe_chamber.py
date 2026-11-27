import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c02_batch_06_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; diag=p['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; s=m._stress_calibration(d['calibration_load_cells']); xt,xe=m._standardize(tr,te)
T,H=m._fit(m._design(tr,xt,s,dt,t),tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('taus weights',T,H)
X=diag['coupon_features_raw'].copy(); lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.r_[tr['ref_coupons'],te['ref_coupons']]
for l in np.unique(lab):
 mask=diag['lab_id']==l; R=ref[lab==l]; a=np.nanmedian(np.nanmean(R,axis=0)); b=np.nanmedian(np.nanstd(R,axis=0)); X[mask]=(X[mask]-a)/b; print('ref lab',l,'train+test',a,b,'diag',np.nanmedian(np.nanmean(diag['ref_coupons'][mask],axis=0)),np.nanmedian(np.nanstd(diag['ref_coupons'][mask],axis=0)))
_,i,n=np.unique(np.c_[lab,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); allx=np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; w=(1/n[i])[:,None]; X=np.where(np.isfinite(X),X,np.sum(w*np.r_[xt,xe]*np.isfinite(allx),axis=0)/np.sum(w*np.isfinite(allx),axis=0)); print('diag missing',np.isnan(diag['coupon_features_raw']).sum(axis=0))
Y=diag['strain_curves_obs']-diag['strain_0'][:,None]; V=diag['instant_rate_obs']; drift=H[2]+X@H[3:]; past,hold=s(diag['past_channel_raw']),s(diag['hold_channel_raw']); print('stress range',past.min(),past.max(),hold.min(),hold.max()); kernel=-np.expm1(-t[:,None]/T); W=-np.expm1(-dt[:,None]/T)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/T); z=past@W
for wait,a,b,g in [(0,1,0,1),(0,0,1,1),(7.2,.22,.79,1),(100,.22,.79,1)]:
 A=(hold[:,None]-z*np.exp(-wait/T))*H[:2]; S=A@kernel.T+drift[:,None]*(a*t-b*g*np.expm1(-t/g)); R=A@(1/T)+(a+b)*drift
 print('archived',wait,a,b,'abs error',np.quantile(abs(S-Y),[.5,.9,.99]),np.quantile(abs(R-V),[.5,.9,1]),'strain rmse by time',np.round(np.sqrt(np.mean((S-Y)**2,axis=0)),3))
np.savez('/tmp/chamber_inputs.npz',T=T,H=H,X=X,lab=diag['lab_id'],Y=Y,V=V,drift=drift,past=past,hold=hold,t=t,dt=dt,kernel=kernel,W=W,z=z)
# Observe whether model error is associated with history, holding load, or intrinsic drift.
raw=np.c_[Y,V]; P=np.c_[np.ones(len(X)),X,hold,past]; B=np.linalg.lstsq(P,raw,rcond=None)[0]; print('per-time unconstrained hold coefficients',np.round(B[5],4)); print('per-time history coefficients',np.round(B[6:],3)); print('per-time drift-feature coefficients',np.round(B[1:5],3)); print('per-time unexplained',np.round(np.linalg.norm(raw-P@B,axis=0),3))
