import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_w04_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_w04_s00.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']
dt,t=d['past_dt'],d['eval_times']; a=m._calibrate(d['calibration_load_cells']); x,_=m._align(tr,te); h,q=m._stress(tr,a)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for method in ('all','complete'):
 print('METHOD',method)
 for lab in range(4):
  val=tr['lab_id']==lab; fit=~val if method=='all' else ~val & complete
  tau,w=m._fit(dt,t,h[fit],q[fit],x[fit],y[fit],r[fit]); u,v=m._design(tau,dt,t,h[val],q[val],x[val]); e=u@w-y[val]; f=v@w-r[val]; c=complete[val]
  print(lab,'taus',np.round(tau,4),'complete early/late abs MAE',*[round(np.median(abs(e[c,j])),6) for j in (slice(None,3),slice(-3,None))],'complete rate MAE',round(np.median(abs(f[c])),6),'missing rate MAE',round(np.median(abs(f[~c])),6) if np.any(~c) else None)
 ix=np.ones(len(y),bool) if method=='all' else complete
 tau,w=m._fit(dt,t,h[ix],q[ix],x[ix],y[ix],r[ix]); print('full fit',np.round(tau,6),np.round(w,6))
