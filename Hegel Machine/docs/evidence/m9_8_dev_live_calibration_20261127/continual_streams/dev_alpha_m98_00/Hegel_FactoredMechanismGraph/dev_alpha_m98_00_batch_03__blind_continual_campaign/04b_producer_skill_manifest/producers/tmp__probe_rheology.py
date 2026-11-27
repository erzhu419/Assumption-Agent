import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py'
s=importlib.util.spec_from_file_location('s',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; print('diagnostics',[(k,np.shape(v)) for k,v in q.items()]); dt,t=d['past_dt'],d['eval_times']; tr=d['train']; a,b=m._prepare(d); tau,th=m._fit(dt,t,a,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
coef=m._stress_calibration(d['calibration_load_cells']); stress=lambda v:v@coef[:4]+coef[4]
print('cal residual',np.sqrt(np.mean((stress(d['calibration_load_cells']['bridge_raw'])-d['calibration_load_cells']['true_channel_loads'].sum(-1))**2)))
x=m._aligned_features(tr,q)[len(tr['lab_id']):]; past=stress(q['past_channel_raw']); hold=stress(q['hold_channel_raw']); z=np.zeros((len(hold),2))
for alpha,st in zip(-np.expm1(-dt[:,None]/tau),past.T): z+=alpha*(st[:,None]-z)
y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; drift=x@th[2:]; A,R=m._design(tau,dt,t,(past,hold,x)); C,S=m._chamber_response(tau,t,hold,x,th)
for name,pc,pr in [('bench',A@th,R@th),('reset',C,S)]: print(name,'curve/rate rmse',np.mean((pc-y)**2)**.5,np.mean((pr-r)**2)**.5)
g=np.c_[-np.expm1(-t[:,None]/tau),t]; rates=np.r_[1/tau,1.]; amps=np.c_[y,r]@np.linalg.pinv(np.vstack([g,rates]).T); print('fixed basis rmse',np.mean((amps@g.T-y)**2)**.5,np.mean((amps@rates-r)**2)**.5)
print('amplitudes: hold weighted, memory weighted, drift, actual coefficients, telemetry')
print(np.c_[hold[:,None]*th[:2],z*th[:2],drift,amps].round(6))
print('columns: weighted hold, weighted memory, drift, projected amplitudes')
for mode in ('train+diag','all','train+test reference'):
 te=d['test']; n=len(tr['lab_id']); nd=len(q['lab_id'])
 if mode=='train+diag': xx=x
 else:
  cat=lambda a,b:{k:np.concatenate((a[k],b[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
  if mode=='all': xx=m._aligned_features(tr,cat(te,q))[-nd:]
  else:
   both=cat(tr,te); aligned=m._aligned_features(tr,te)[:,1:]; raw=q['coupon_features_raw'].copy()
   for lab in np.unique(q['lab_id']):
    pan=both['ref_coupons'][both['lab_id']==lab]; rows=q['lab_id']==lab
    raw[rows]=(raw[rows]-np.median(np.nanmean(pan,axis=0)))/np.median(np.nanstd(pan,axis=0))
   w=1/np.unique(np.c_[both['lab_id'],both['stratum']],axis=0,return_inverse=True,return_counts=True)[2][np.unique(np.c_[both['lab_id'],both['stratum']],axis=0,return_inverse=True)[1]]
   miss=np.isnan(both['coupon_features_raw']); fill=np.sum(w[:,None]*np.where(miss,0,aligned),axis=0)/np.sum(w[:,None]*~miss,axis=0); xx=np.c_[np.ones(nd),np.where(np.isnan(raw),fill,raw)]
 dd=xx@th[2:]; print(mode,'drift range',dd.min(),dd.max())
 bases=[np.c_[hold*th[0],z[:,0]*th[0],dd],np.c_[hold*th[1],z[:,1]*th[1],dd],np.c_[dd,dd**2, np.ones(nd)]]
 for b,B in enumerate(bases):
  zfit=np.linalg.lstsq(B,amps[:,b],rcond=None)[0]; print(b,zfit.round(5),'amplitude rmse',np.mean((B@zfit-amps[:,b])**2)**.5)
 H=np.stack([g[:,b][None,:,None]*B[:,None,:] for b,B in enumerate(bases)],axis=2).transpose(0,1,2,3).reshape(nd,len(t),9)
 J=np.column_stack([rates[b]*B for b,B in enumerate(bases)])
 M=np.vstack([H.reshape(-1,9), J]); Y=np.r_[y.ravel(),r]
 fit=np.linalg.lstsq(M,Y,rcond=None)[0]; print('direct linear',fit.reshape(3,3).round(5),'rmse',np.sqrt(np.mean((M@fit-Y)**2)))
 print('resid by time',np.sqrt(np.mean((H@fit-y)**2,axis=0)).round(5),'rate',np.sqrt(np.mean((J@fit-r)**2)))
