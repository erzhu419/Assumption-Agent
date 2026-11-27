import sys,pickle,numpy as np,importlib.util
p='/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py'; s=importlib.util.spec_from_file_location('a',p); a=importlib.util.module_from_spec(s); s.loader.exec_module(a)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; g=it['campaign_diagnostics']; tr,te=d['train'],d['test']; c=d['calibration_load_cells']; t,dt=d['eval_times'],d['past_dt']; b=a._bridge(c); n=len(tr['lab_id']); x=a._features(tr,te); p,h=a._loads(tr,b); tau,th=a._fit(x[:n],p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau,theta',tau,th)
# Extend the same reference-panel transform to diagnostic coupons without changing the material fit.
lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); xg=g['coupon_features_raw'].copy()
for l in np.unique(lab):
 q=ref[lab==l]; j=g['lab_id']==l; xg[j]=(xg[j]-np.median(q.mean(0)))/np.median(q.std(0))
for j in range(4):
 m=~np.isfinite(xg[:,j]); m0=~np.isfinite(raw[:,j]); xg[m,j]=np.median(x[m0,j]) if np.any(m0) else 0
pg,hg=a._loads(g,b); yc=g['strain_curves_obs']-g['strain_0'][:,None]; yr=g['instant_rate_obs']; ug=g.get('aux_chamber_telemetry'); A,R=a._design(xg,pg,hg,dt,t,tau); base=(A@th).reshape(yc.shape); rbase=R@th
print('diag telemetry',None if ug is None else ug.shape,'base strain and rate rmse',np.sqrt(np.mean((yc-base)**2)),np.sqrt(np.mean((yr-rbase)**2)))
F=np.column_stack([-np.expm1(-t[:,None]/tau),t]); C=np.linalg.lstsq(F,yc.T,rcond=None)[0].T; Cb=np.linalg.lstsq(F,base.T,rcond=None)[0].T
print('SVD shape residual',np.linalg.svd(yc-C@F.T,compute_uv=False)); print('projected rate residual rms',np.std(yr-C@np.r_[1/tau,1])); print('obs coef/base coef/hold/lab',np.round(np.c_[C,Cb,hg,g['lab_id']],4)); print('curve residual median by time',np.round(np.median(yc-base,axis=0),5))
