import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_s00_solver.py'; s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; train,test,diag=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; fit=m.solve_batch(d); tau,w=fit['estimated_taus'],fit['hereditary_weights']; print('tau,w',tau,w)
cal=m._stress_calibration(d['calibration_load_cells']); print('cal',cal); c=d['calibration_load_cells']; print('cal residual quantiles',np.quantile(c['bridge_raw']@cal[:-1]+cal[-1]-c['true_channel_loads'].sum(1),[0,.25,.5,.75,1])); merged={k:np.concatenate((test[k],diag[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; x=m._aligned_features(train,merged)
for g,x1 in ((train,x[:len(train['lab_id'])]),(diag,x[-len(diag['lab_id']):])):
 past,hold=g['past_channel_raw']@cal[:-1]+cal[-1],g['hold_channel_raw']@cal[:-1]+cal[-1]; A,B=m._design(tau,dt,t,(past,hold,x1)); e=g['strain_curves_obs']-g['strain_0'][:,None]-A@w; r=g['instant_rate_obs']-B@w
 print('curve by time',np.round(np.quantile(e,[.1,.5,.9],axis=0),5)); print('rate',np.round(np.quantile(r,[0,.1,.5,.9,1]),5))
 G=np.vstack((np.column_stack((-np.expm1(-t[:,None]/tau),t)),np.r_[1/tau,1.])); delta=np.linalg.lstsq(G,np.column_stack((e,r)).T,rcond=None)[0].T; ep=np.column_stack((e,r))-delta@G.T; print('projection error',np.quantile(np.abs(ep),[.5,.9,1]));
 if g is diag:
  print('lab,stratum,hold,past[6],drift,rate delta,coef delta[3],projection maxabs'); print(np.round(np.column_stack((g['lab_id'],g['stratum'],hold,past,x1@w[2:],r,delta,np.max(abs(ep),axis=1))),5))
