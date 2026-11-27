import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_06_solver.py'
s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; a,b=c['bridge_raw'],c['true_channel_loads'];
def groups(y):
 return np.percentile(np.max(np.abs(y),axis=1),[0,50,75,90,95,98,100]).round(6)
for deg in range(1,5):
 from itertools import combinations_with_replacement
 A=np.column_stack([np.prod(b[:,idx],axis=1) for k in range(deg+1) for idx in combinations_with_replacement(range(4),k)])
 mask=np.max(np.abs(a),axis=1)<4*np.median(np.max(np.abs(a),axis=1));
 for j in range(4):
  co=np.linalg.lstsq(A[mask],a[mask],rcond=None)[0]; y=A@co-a; e=np.max(np.abs(y),axis=1); mask=e<max(12*np.median(e),1e-8)
 print('full degree',deg,'inliers',sum(mask),'residual percentiles',groups(y))
x,_=m._standardize(tr,te); combined={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}; _,xd=m._standardize(combined,di)
f=m._stress_calibration(c); M=m._design(tr,x,f,dt,t); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; taus,th=m._fit(M,y,r); A,B=M(taus); print('taus theta',taus,th,'train median',np.median(abs(A@th-y)),np.median(abs(B@th-r)))
h=f(di['hold_channel_raw']); q=f(di['past_channel_raw']); drift=th[2]+xd@th[3:]; tail=np.cumsum(dt[::-1])[::-1]-dt; z=q@(-np.expm1(-dt[:,None]/taus)*np.exp(-tail[:,None]/taus)); yd=di['strain_curves_obs']-di['strain_0'][:,None]; rd=di['instant_rate_obs']; R=-np.expm1(-t[:,None]/taus)*th[:2]; g=th[:2]/taus
for name,C in [('linear',t),('log',np.log1p(t)),('sat',-np.expm1(-t))]:
 for name2,W in [('keep',np.ones(2)),('reset',np.zeros(2)),('keep0',[1,0]),('keep1',[0,1])]:
  gap=h[:,None]-z*W; e=gap@R.T+drift[:,None]*C-yd; er=gap@g+drift-rd; print(name,name2,'median',np.median(abs(e)).round(6),np.median(abs(er)).round(6),'late',np.median(abs(e[:,-3:])).round(6))
for n in ['train','test']:
 a=d[n]['aux_chamber_telemetry']; print(n,'telemetry',a.mean(0),a.std(0))
