import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m98_00_s00_solver.py'); a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; ptrain,ptest=a._prepare(d)
two=lambda k:np.concatenate([tr[k],te[k]])
lab=two('lab_id'); st=two('stratum'); panel=two('ref_coupons'); raw=two('coupon_features_raw').copy(); x=q['coupon_features_raw'].copy()
for l in np.unique(lab):
 m=lab==l;r=panel[m]; c=np.median(np.nanmean(r,axis=0)); f=np.median(np.nanstd(r,axis=0)); raw[m]=(raw[m]-c)/f; x[q['lab_id']==l]=(x[q['lab_id']==l]-c)/f
_,inv,cnt=np.unique(np.column_stack([lab,st]),axis=0,return_inverse=True,return_counts=True); w=1./cnt[inv,None]; x=np.where(np.isnan(x),np.nansum(raw*w,axis=0)/np.sum(w*np.isfinite(raw),axis=0),x)
cal=a._stress_calibration(d['calibration_load_cells']); pq=(q['past_channel_raw']@cal[:-1]+cal[-1],q['hold_channel_raw']@cal[:-1]+cal[-1],np.column_stack((np.ones(len(x)),x)))
out=a.solve_batch(d); tau=out['estimated_taus']; b=out['hereditary_weights']; print('tau theta:',tau,b,'dt:',dt)
for label,g,p in [('bench',tr,ptrain),('chamber',q,pq)]:
 c,r=a._design(tau,dt,t,p); y=g['strain_curves_obs']-g['strain_0'][:,None]; v=g['instant_rate_obs']; e=y-c@b; er=v-r@b
 print(label,'curve RMSE',np.sqrt(np.mean(e**2)),'rate RMSE',np.sqrt(np.mean(er**2)),'median curve residual',np.round(np.median(e,axis=0),5),'rate range',np.round(np.percentile(er,[0,25,50,75,100]),5))
 A=np.column_stack((-np.expm1(-t[:,None]/tau),t)); B=np.row_stack((A,[* (1/tau),1.])); D=np.linalg.lstsq(B,np.column_stack((e,er)).T,rcond=None)[0].T; R=np.column_stack((e,er))-D@B.T
 print('branch1 branch2 drift residual coefficients; row fit RMSE',np.round(np.column_stack((D,np.sqrt(np.mean(R**2,axis=1))))[:18],5)); print('telemetry', np.round(g.get('aux_chamber_telemetry',np.zeros((len(y),0)))[:4],4))
 if label=='chamber':
  past,h,xx=p; feat=np.column_stack((past,h,xx[:,1:],c[:,-1,:2])); print('correlations [past0..5 hold x0..3 gap0 gap1]\n',np.round(np.corrcoef(np.column_stack((D,feat)).T)[:3,3:],3)); print('curves residual rows:\n',np.round(e[:6],5))
