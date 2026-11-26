import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w02_batch_07_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_w02_batch_07.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; labs=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; f=s._features(tr,te); c=s._calibrate(d['calibration_load_cells']); x=s._inputs(tr,c,dt,d['yield_stress'],f[:len(labs)]); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
for lab in np.unique(labs):
 m=labs==lab; tau,w=s._fit(tuple(a[~m] for a in x),dt,t,y[~m]); e=s._design(tuple(a[m] for a in x),dt,t,np.log(tau))@w-y[m]; a=abs(e[:,:-1]); raw=tr['coupon_features_raw'][m]; print('lab',lab,'abs strain median/p90/p99',np.round(np.quantile(a,[.5,.9,.99]),4),'rate median/p90',np.round(np.quantile(abs(e[:,-1]),[.5,.9]),4))
 print('  curve noise via residual second slopes median/p90',np.round(np.quantile(abs(np.diff(np.diff(e[:,:-1],axis=1)/np.diff(t),axis=1)),[.5,.9]),4),'features missing',np.sum(~np.isfinite(raw),axis=0),'spread',np.round(np.std(f[:len(labs)][m],axis=0),3))
 for label,idx in [('complete',np.isfinite(raw).all(axis=1)),('imputed',~np.isfinite(raw).all(axis=1))]:
  if idx.any(): print(' ',label,int(idx.sum()),'strain median AE',round(np.median(a[idx]),5),'rate median AE',round(np.median(abs(e[idx,-1])),5),'max horizon median signed',round(np.max(abs(np.median(e[idx,:-1],axis=0))),5))
