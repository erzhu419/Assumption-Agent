import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'
s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; train,test=m._prepare(d)
xall=m._features(tr,te); x=c['coupon_features_raw'].copy(); lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack((tr['ref_coupons'],te['ref_coupons'])); st=np.r_[tr['stratum'],te['stratum']]; _,inv,cnt=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1/cnt[inv]; rx=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])).copy()
for j in np.unique(lab):
 r=ref[lab==j]; center=np.median(r.mean(0)); scale=np.median(r.std(0)); x[c['lab_id']==j]=(x[c['lab_id']==j]-center)/scale; rx[lab==j]=(rx[lab==j]-center)/scale
fill=np.nansum(w[:,None]*rx,axis=0)/(w[:,None]*np.isfinite(rx)).sum(0); x=np.where(np.isfinite(x),x,fill)
o,b=m._calibration(d['calibration_load_cells']); past=(c['past_channel_raw']-o)@b; hold=(c['hold_channel_raw']-o)@b; feat=np.c_[np.ones(len(x)),x]; ch=(past,hold,feat)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; v=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; taus,theta=m._fit(train,dt,t,y); drift=feat@theta[2:]; E=-np.expm1(-t[:,None]/taus)
print('parameters',taus,theta); print('cal residual rms',np.sqrt(np.mean(((d['calibration_load_cells']['bridge_raw']-o)@b-d['calibration_load_cells']['true_channel_loads'].sum(1))**2)))
def metrics(name,pred):
 r=pred-v; print(name, 'RMSE',np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2)),'medAE',np.median(abs(r[:,:-1])),np.median(abs(r[:,-1])))
base=m._design(ch,dt,t,taus)@theta; mem=hold[:,None]-m._design(ch,dt,np.array([1.]),taus)[:,-1,:2]*taus
reset=np.c_[hold[:,None]*(E@theta[:2])+drift[:,None]*t,hold*np.sum(theta[:2]/taus)+drift]
arch=np.c_[(hold[:,None]*theta[:2]+drift[:,None]*taus*[.41,.59])@E.T,reset[:,-1]]
metrics('benchtop',base);metrics('reset-linear',reset);metrics('archived reset-modal',arch)
print('per-time free regression against stress/nominal memory/drift'); A=np.c_[hold,mem,drift,np.ones(len(x))]; coef=np.linalg.lstsq(A,v,rcond=None)[0]; print(np.round(coef,4)); print('expected hold',np.round(np.r_[E@theta[:2],np.sum(theta[:2]/taus)],4)); print('nominal memory',np.round(np.vstack((E*theta[:2],theta[:2]/taus)).T,4));print('drift residual temporal shape',np.round(np.linalg.lstsq(drift[:,None],v-base,rcond=None)[0],4)); metrics('free',A@coef)
print('raw train residual',np.median(abs(m._design(train,dt,t,taus)@theta-y),axis=0)); print('diag feature missing count',np.isnan(c['coupon_features_raw']).sum())
