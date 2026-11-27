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
