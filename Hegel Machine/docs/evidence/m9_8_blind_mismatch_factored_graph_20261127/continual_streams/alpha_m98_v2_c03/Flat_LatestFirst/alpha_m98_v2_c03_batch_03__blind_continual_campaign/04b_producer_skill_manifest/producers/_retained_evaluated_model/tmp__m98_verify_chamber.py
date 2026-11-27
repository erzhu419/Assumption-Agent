import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; out=s.solve_batch(d); T=out['estimated_taus']; w=out['hereditary_weights']; t=d['eval_times']; dt=d['past_dt']; o,c=s._calibration(d['calibration_load_cells']); print('material',T,w)
# The diagnostic cohort was standardized independently. In each fold,
# estimate its missing-value means from non-held-out diagnostic labs.
x=g['coupon_features_raw'].copy(); lab=g['lab_id']; st=g['stratum'];
for l in np.unique(lab):
 m=lab==l; a=g['ref_coupons'][m]; x[m]=(x[m]-np.median(a.mean(0)))/np.median(a.std(0))
y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; p=(g['past_channel_raw']-o)@c; h=(g['hold_channel_raw']-o)@c
for l in np.unique(lab):
 m=lab==l; j=~m; _,inv,n=np.unique(np.c_[lab[j],st[j]],axis=0,return_inverse=True,return_counts=True); weights=1/n[inv].reshape(-1,1); mu=np.nansum(weights*x[j],0)/np.sum(weights*~np.isnan(x[j]),0); f=np.where(np.isnan(x[m]),mu,x[m]); X=np.c_[np.ones(sum(m)),f]; A=s._design((p[m],h[m],X),dt,t,T); e=A[:,:,:2]@w[:2]+(X@w[2:])[:,None]*np.r_[-np.expm1(-t),1.]-y[m]; print('lab',l,'curve/final/rate RMSE',np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-2]**2),np.mean(e[:,-1]**2)]),'max-horizon RMSE',np.sqrt(np.mean(e[:,:-1]**2,axis=0)).max())
