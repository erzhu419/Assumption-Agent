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
g=np.c_[-np.expm1(-t[:,None]/tau),t]; rates=np.r_[1/tau,1.]; n=len(hold)
print('sizes',len(tr['lab_id']),len(d['test']['lab_id']),n)
cat=lambda a,b:{k:np.concatenate((a[k],b[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
empty={k:v[:0] for k,v in cat(tr,tr).items()}; te=d['test']
for label, s in [('train+diag',tr),('diag only',empty),('test+diag',te),('all',cat(tr,te))]:
 xx=m._aligned_features(s,q)[-n:]; dd=xx@th[2:]; h=hold[:,None]*th[:2]; zm=z*th[:2]; gap=h-zm; obs=np.c_[y,r]
 print('\n',label, 'missing rows:', np.argwhere(np.isnan(q['coupon_features_raw'])).tolist()); print('projected', np.c_[dd, amps[:,0]-gap[:,0], amps[:,1]-gap[:,1], amps[:,2]].round(4).T)
 def measure(blocks,offset, name):
  F=np.concatenate([G[:,b][None,:,None]*B[:,None,:] for b,B in enumerate(blocks)],axis=2); Y=obs-offset; p=F.shape[-1]
  beta=np.linalg.lstsq(F.reshape(-1,p),Y.ravel(),rcond=None)[0]; pred=np.empty_like(Y)
  for j in range(n):
   I=np.arange(n)!=j; pred[j]=F[j]@np.linalg.lstsq(F[I].reshape(-1,p),Y[I].ravel(),rcond=None)[0]
  ae=pred-Y; print(name,'beta',beta.round(3),'loo curve/rate/worst',np.round([np.sqrt(np.mean(ae[:,:-1]**2)),np.sqrt(np.mean(ae[:,-1]**2)),np.max(np.sqrt(np.mean(ae**2,axis=1)))],4))
 D=dd[:,None]; G=np.vstack([g,rates]); off=gap@G[:,:2].T
 for nm,B in [('linear',D),('signed abs',np.c_[D,abs(D)]),('quadratic',np.c_[D,D**2]),('cubic',np.c_[D,D**2,D**3]),('raw feature',xx)]: measure([B]*3,off,nm)
 measure([np.c_[h[:,0],zm[:,0],D],np.c_[h[:,1],zm[:,1],D],D],0,'free gap')
 measure([np.c_[h[:,0],zm,D],np.c_[h[:,1],zm,D],D],0,'cross memory')
print('raw features + lab/stratum'); print(np.c_[q['lab_id'],q['stratum'],q['coupon_features_raw']].T)
