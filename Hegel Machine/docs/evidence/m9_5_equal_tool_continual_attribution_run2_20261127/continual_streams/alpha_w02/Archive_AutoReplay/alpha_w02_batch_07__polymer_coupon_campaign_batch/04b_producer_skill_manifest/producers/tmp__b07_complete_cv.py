import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w02_batch_07_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_w02_batch_07.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; labs=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; f=s._features(tr,te); cal=s._calibrate(d['calibration_load_cells']); x=s._inputs(tr,cal,dt,d['yield_stress'],f[:len(labs)]); z=s._inputs(te,cal,dt,d['yield_stress'],f[len(labs):]); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for name, keep in [('all',np.ones_like(complete)),('complete_fit',complete)]:
 E=np.empty_like(y)
 for lab in np.unique(labs):
  m=labs==lab; use=keep&~m; tau,w=s._fit(tuple(a[use] for a in x),dt,t,y[use]); E[m]=s._design(tuple(a[m] for a in x),dt,t,np.log(tau))@w-y[m]
  print(name,lab,'complete early/late strain and onset MedAE',np.round([np.median(abs(a)) for a in np.split(E[m&complete,:-1],2,axis=1)]+[np.median(abs(E[m&complete,-1]))],5),'complete max horizon MAE',round(np.mean(abs(E[m&complete,:-1]),axis=0).max(),5),'imputed strain MedAE',round(np.median(abs(E[m&~complete,:-1])),5))
 tau,w=s._fit(tuple(a[keep] for a in x),dt,t,y[keep]); pred=s._design(z,dt,t,np.log(tau))@w
 if name=='all': base=pred
 print(name,'pooled complete strain/rate MedAE',np.median(abs(E[complete,:-1])),np.median(abs(E[complete,-1])),'taus',tau,'weights',w[:7],'test max strain/rate shift',np.max(abs(pred[:,:-1]-base[:,:-1])), np.max(abs(pred[:,-1]-base[:,-1])))
