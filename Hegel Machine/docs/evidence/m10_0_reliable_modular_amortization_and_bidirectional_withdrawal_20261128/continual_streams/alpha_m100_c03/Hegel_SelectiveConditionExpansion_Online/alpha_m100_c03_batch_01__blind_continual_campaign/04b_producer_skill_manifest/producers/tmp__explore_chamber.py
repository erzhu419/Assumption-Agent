import pickle,runpy,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; dg=p['campaign_diagnostics']; c=d['calibration_load_cells']; t=d['eval_times']; dt=d['past_dt']; m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c03_s00_solver.py'); np.set_printoptions(precision=5,suppress=True,linewidth=130)
for a,y in [(c['bridge_raw'],c['true_channel_loads']),(c['true_channel_loads'],c['bridge_raw'])]:
 for deg in [1,2,3]:
  from itertools import combinations_with_replacement
  X=np.column_stack([np.ones(len(a))]+[np.prod(a[:,cols],axis=1) for k in range(1,deg+1) for cols in combinations_with_replacement(range(4),k)])
  H=X@np.linalg.pinv(X); R=(H@y-y)/(1-H.diagonal()[:,None]); print('cal degree',deg,'LOO rmse/std',np.sqrt(np.mean(R**2,axis=0))/y.std(axis=0),'sum rmse',np.sqrt(np.mean(R.sum(1)**2)))
b=m['_bridge'](c); x=m['_features'](tr,te); z=m['_features'](tr,dg)[len(tr['lab_id']):]; past,hold=m['_loads'](tr,b); tau,w=m['_fit'](x[:len(hold)],past,hold,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau,theta',tau,w)
for split,X in [(tr,x[:len(hold)]),(dg,z)]:
 past,hold=m['_loads'](split,b); C,R=m['_design'](tau,X,past,hold,dt,t); Y=split['strain_curves_obs']-split['strain_0'][:,None]; V=split['instant_rate_obs']; E=Y-C@w; print('lab,stratum',np.c_[split['lab_id'],split['stratum']][:24].T); print('curve and rate abs residual quantiles',np.quantile(abs(E),[.5,.9,1]),np.quantile(abs(V-R@w),[.5,.9,1])); print('resid sv',np.linalg.svd(E,compute_uv=False)); print('hold range',np.quantile(hold,[0,.5,1]));
 if split is dg:
  A=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t,t*t],np.r_[1/tau,1,0])); W=np.vstack((E.T,V-R@w)); co=np.linalg.lstsq(A,W,rcond=None)[0].T; print('chamber coefficient columns (two amps, slope, quad)',co); print('amplitude-fit abs residual quantiles',np.quantile(abs(W-A@co.T),[.5,.9,1])); print('predictors: hold last avg base_drift x0..x3'); print(np.c_[hold,past[:,-1],past.mean(1),R[:,2:]@w[2:],X]); print('corr',np.corrcoef(np.c_[co,hold,past,R[:,2:]@w[2:],X].T)[:4,4:]); print('strain residual rows',E[:5]); print('rate residual',V-R@w)
print('aux train,test ranges',[(a['aux_chamber_telemetry'].min(0),a['aux_chamber_telemetry'].max(0),a['aux_chamber_telemetry'][:3]) for a in (tr,te)]); print('sweep head', {k:v[:12] for k,v in c['single_channel_sweeps'].items()})
