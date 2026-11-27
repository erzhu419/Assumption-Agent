import pickle, numpy as np, importlib.util
p='/tmp/hegel_continual_archive/alpha_m98_v2_c01_batch_06_solver.py'; spec=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; inv,_=m.fit_bridge(d['calibration_load_cells']); xtr,xte=m._features(tr,te); tau,w=m._fit(m._specimen(tr,xtr,inv),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);
merged={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; _,xg=m._features(merged,g); _,xalt=m._features(g,g); xref=np.empty_like(xg)
for lab in np.unique(g['lab_id']):
 mask=g['lab_id']==lab; ref=merged['ref_coupons'][merged['lab_id']==lab]; xref[mask]=(g['coupon_features_raw'][mask]-np.median(ref.mean(0)))/np.median(ref.std(0))
print('missing diag',np.isnan(xref).sum(0)); xref=np.where(np.isfinite(xref),xref,xg)
past,h,_=m._specimen(g,xg,inv); z=np.zeros((len(h),2));
for j,delta in enumerate(dt):
 f=np.exp(-delta/tau); z=f*z+(1-f)*past[:,j,None]
E=-np.expm1(-t[:,None]/tau); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; driftlaws={'linear':t,'log':np.log1p(t),'exp':-np.expm1(-t),'rational':t/(1+t),'saturate2':-np.expm1(-2*t)/2};
for xn,x in [('joint',xg),('diag',xalt),('reference',xref)]:
 drift=w[2]+x@w[3:]; ranked=[]
 for mem,Z in [('history',z),('reset',np.zeros_like(z)),('reverse',z[:,::-1])]:
  A=(h[:,None]-Z)*w[:2]; er=np.abs(A@(1/tau)+drift-r)
  for name,F in driftlaws.items():
   es=np.abs(A@E.T+drift[:,None]*F-y); ranked.append((np.median(es),np.quantile(es,.9),np.median(er),np.quantile(er,.9),mem,name))
 print(xn,'top',sorted(ranked)[:6])
 for mem,Z in [('history',z),('reset',np.zeros_like(z))]:
  A=(h[:,None]-Z)*w[:2]; residual=y-A@E.T; ratio=(drift@residual)/(drift@drift); print(xn,mem,'profile',np.round(ratio,4),'rate discrepancy',np.round(r-A@(1/tau)-drift,3))
print('hold range',h.min(),h.max(),'history range',z.min(0),z.max(0))
