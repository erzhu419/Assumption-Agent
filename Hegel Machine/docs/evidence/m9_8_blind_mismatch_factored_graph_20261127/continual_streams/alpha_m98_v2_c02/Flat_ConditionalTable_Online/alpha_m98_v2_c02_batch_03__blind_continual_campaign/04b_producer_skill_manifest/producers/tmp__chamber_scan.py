import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,di=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; stress=m._stress_calibration(d['calibration_load_cells']); xtr,xte=m._standardize(tr,te); _,xdi=m._standardize(tr,di); taus,theta=m._fit(m._design(tr,xtr,stress,dt,t),tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('TAUS',taus,'THETA',theta,'XDI MISSING',np.isnan(di['coupon_features_raw']).sum(0)); print('STRESS RANGES',np.ptp(stress(di['hold_channel_raw'])),np.ptp(stress(di['past_channel_raw']))); print('REF',np.array([np.median(a['ref_coupons'],axis=0) for a in (tr,te,di)]));
y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; hold=stress(di['hold_channel_raw']); past=stress(di['past_channel_raw']); drift=theta[2]+xdi@theta[3:]; decay=-np.expm1(-t[:,None]/taus); z=np.zeros((len(hold),2)); zr=z.copy()
for i in range(len(dt)):
 f=-np.expm1(-dt[i]/taus); z+=(past[:,i,None]-z)*f; zr+=(past[:,-1-i,None]-zr)*f
for name,gap in [('normal',hold[:,None]-z),('reset',np.repeat(hold[:,None],2,axis=1)),('reverse',hold[:,None]-zr),('last',np.repeat((hold-past[:,-1])[:,None],2,axis=1)),('first',np.repeat((hold-past[:,0])[:,None],2,axis=1))]:
 for nm,tf in [('linear',t),('log',np.log1p(t)),('sat',-np.expm1(-t)),('zero',0*t)]:
  e=y-(gap*theta[:2])@decay.T-drift[:,None]*tf; er=r-(gap*theta[:2]/taus).sum(1)-drift*(nm!='zero'); print(name,nm,'MAE',np.mean(abs(e)),np.mean(abs(er)),'median',np.median(abs(e)),np.median(abs(er)))
print('sample first / last / rate');print(np.round(np.c_[hold,z,drift,y[:,0],y[:,-1],r],5))
