import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; train,diag=m._prepare(dict(d,test=c)); y=np.column_stack([tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]); tau,theta=m._fit(train,dt,t,y)
yc=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); pred=m._design(diag,dt,t,tau)@theta; r=yc-pred; past,h,f=diag; drift=f@theta[2:]; g=m._design(diag,dt,t,tau)[:,-1,:2]*tau; z=h[:,None]-g
rm=lambda a:np.sqrt(np.mean(a*a,axis=0))
lab=c['lab_id']; bins=np.unique(lab); actual=m._prepare(d)[0]; allx=m._features(tr,{k:np.concatenate([d['test'][k],c[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}); print('relative feature normalization shift max/median',np.max(abs(allx[-len(h):]-f[:,1:])),np.median(abs(allx[-len(h):]-f[:,1:]))); print('material fit on actual input',m._fit(actual,dt,t,y)); print('missing diagnostic',np.argwhere(np.isnan(c['coupon_features_raw'])).tolist()); print('times+rate',np.r_[t,0].round(3));
for name,A in [('drift',np.c_[np.ones(len(h)),drift]),('stress+drift',np.c_[np.ones(len(h)),drift,h,z]),('plus features',np.c_[np.ones(len(h)),f[:,1:],h,z]),('plus labs',np.c_[np.ones(len(h)),drift,h,z,lab[:,None]==bins[1:]]),('plus final history',np.c_[np.ones(len(h)),drift,h,z,past[:,-1]])]:
 p=np.linalg.lstsq(A,r,rcond=None)[0]; print(name,'coefs\n',p.round(4),'rmse',rm(r-A@p).round(4))
 for q in [lab==L for L in bins]:
  est=A[q]@np.linalg.lstsq(A[~q],r[~q],rcond=None)[0]; print(' held lab',lab[q][0],np.sqrt(np.mean((est[:,:-1]-r[q,:-1])**2)).round(4),rm(est[:,-1]-r[q,-1]).round(4))
print('sweep stats',[(k,np.shape(v)) for k,v in d['calibration_load_cells']['single_channel_sweeps'].items()])
cal=d['calibration_load_cells']; loads=cal['true_channel_loads']; raw=cal['bridge_raw']; A=np.c_[np.ones(len(loads)),loads]; fit=np.linalg.lstsq(A,raw,rcond=None)[0]; print('cal rms',rm(A@fit-raw),'condition',np.linalg.cond(A),'bridge',fit.round(4))
