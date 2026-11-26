import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('m','/holdout_task/alpha_w02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w02_batch_07.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress')); lab=tr['lab_id']; n=len(lab); f=m._features(tr,te)[:n]; inp=m._inputs(tr,f,m._calibrate(d['calibration_load_cells'],sy),dt,sy); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; complete=np.isfinite(tr['coupon_features_raw']).all(1); h=np.r_[t,1.]; r=np.empty_like(y)
for j in range(4):
 fit=complete&(lab!=j); val=lab==j; tau,c=m._fit(tuple(v[fit] for v in inp),dt,t,y[fit]); r[val]=y[val]-m._design(tuple(v[val] for v in inp),dt,t,tau)@c
q=lambda x: np.round(np.quantile(x,[.1,.5,.8,.9,.95,.98,1],axis=0),5)
print('absolute residual quantiles by horizon and rate:',q(abs(r[complete])))
print('median residuals by lab:',*[np.round(np.median(r[complete&(lab==j)],axis=0),5) for j in range(4)])
drift=np.median(r[:,:-1]/t,axis=1); diff=r[:,:-1]-drift[:,None]*t
print('absolute detrended residual quantiles:',q(abs(diff[complete])))
for j in range(4):
 for full in (True,False):
  a=(lab==j)&(complete==full); print('lab/complete',j,full,'n',a.sum(),'abs drift percentiles',q(abs(drift[a])),'abs drift minus rate percentiles',q(abs(drift[a]-r[a,-1])))
print('raw fit gains',m._rate_gains(inp,dt,t,y,lab,complete))
