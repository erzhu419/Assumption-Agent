import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; ch=p['campaign_diagnostics']; o=s.solve_batch(d); tau,w=o['estimated_taus'],o['hereditary_weights']; t=d['eval_times']; dt=d['past_dt']; bridge=s._bridge(d['calibration_load_cells']); past,h=s._loads(ch,bridge)
xc=s._features(ch,{key:ch[key][:0] for key in ('lab_id','stratum','ref_coupons','coupon_features_raw')}); score=w[2]+xc@w[3:]; m=np.c_[np.ones(len(score)),score]; basis=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t], np.r_[1/tau,1.]]); obs=np.c_[ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs']]; adjusted=obs-(h[:,None]*w[:2])@basis[:,:2].T
rms=lambda a,axis=None: np.sqrt(np.mean(a*a,axis=axis))
print('taus/weights',tau,w); assert 0.05<tau[0]<tau[1]<25
print('recovered law',np.linalg.lstsq(m,np.linalg.lstsq(basis,adjusted.T,rcond=None)[0].T,rcond=None)[0]);
max_fold=np.zeros(3)
for key in ('lab_id','stratum'):
 for k in np.unique(ch[key]):
  f=ch[key]==k; law=np.linalg.lstsq(m[~f],np.linalg.lstsq(basis,adjusted[~f].T,rcond=None)[0].T,rcond=None)[0]
  pred,rate=s._chamber(tau,w,h[f],xc[f],t,law); err=obs[f]-np.c_[pred,rate]; stats=np.array([rms(err[:,:-1]),max(rms(err[:,:-1],0)),rms(err[:,-1])]); max_fold=np.maximum(max_fold,stats); print('chamber',key,k,stats)
print('worst group curve, worst time point, worst group rate',max_fold); assert np.all(max_fold<[0.007,0.010,0.006])
calc,rate=s._chamber(tau,w,h,xc,t); e=obs-np.c_[calc,rate]; missing=~np.isfinite(ch['coupon_features_raw']).all(1); print('missing chamber RMSE',rms(e[missing,:-1]),rms(e[missing,-1]))
eps=1e-6; derivative=s._chamber(tau,w,h,xc,np.array([eps]))[0][:,0]/eps; assert max(abs(derivative-rate))<1e-5
for lab in np.unique(d['train']['lab_id']):
 tr,te=d['train'],d['test']; x=s._features(tr,te)[:len(tr['lab_id'])]; take=tr['lab_id']!=lab; ph,hh=s._loads(tr,bridge); q=tr['strain_curves_obs']-tr['strain_0'][:,None]; a,b=s._fit(ph[take],hh[take],x[take],dt,t,q[take],tr['instant_rate_obs'][take]); A,B=s._design(a,ph[~take],hh[~take],x[~take],dt,t); err=q[~take]-A@b; er=tr['instant_rate_obs'][~take]-B@b; print('benchtop held lab',lab,'max relative tau shift',max(abs(a/tau-1)),'max weight shift',max(abs(b-w)),'curve median absolute, rate RMSE',np.median(abs(err)),rms(er)); assert max(abs(a/tau-1))<0.1 and max(abs(b-w))<0.05
