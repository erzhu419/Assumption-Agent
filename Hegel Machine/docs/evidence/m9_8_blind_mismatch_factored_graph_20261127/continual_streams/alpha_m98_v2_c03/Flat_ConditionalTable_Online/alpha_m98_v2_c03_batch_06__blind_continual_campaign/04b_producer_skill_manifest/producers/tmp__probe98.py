import pickle,sys,numpy as np, importlib.util
p='/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_01_solver.py'; s=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; tr,te=m._prepare(d); y=np.column_stack((d['train']['strain_curves_obs']-d['train']['strain_0'][:,None],d['train']['instant_rate_obs'])); tau,th=m._fit(tr,dt,t,y); print('tau, theta, dt,t',tau,th,dt,t); print('train residual MAD',np.median(abs(m._design(tr,dt,t,tau)@th-y),axis=0)); print('shapes',len(tr[0]),len(te[0]),len(g['lab_id'])); print('cal', {k:type(v) for k,v in d['calibration_load_cells'].items()})
o,c=m._calibration(d['calibration_load_cells']); gy=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; past=(g['past_channel_raw']-o)@c; hold=(g['hold_channel_raw']-o)@c

def combine(*groups):
 return {k:np.concatenate([q[k] for q in groups]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
for ref in (g, combine(d['train'],g),combine(d['train'],d['test'],g)):
 f=m._features(ref,ref)[-len(hold):]; cp=past,hold,np.c_[np.ones(len(hold)),f]; a=m._design(cp,dt,t,tau); elastic=a[:,:,:2]@th[:2]; drift=cp[2]@th[2:]; e=lambda u:np.vstack((-np.expm1(-t[:,None]/np.atleast_1d(u)),1/np.atleast_1d(u))); lin=np.r_[t,1.]; exp=e(tau); ws=np.r_[np.repeat(.02,len(t)),.03]; print('normalization rows',len(ref['lab_id']), 'f mean',f.mean(0));
 for mode,base in [('forward',elastic),('reset',hold[:,None]*(exp@th[:2])),('reverse',(m._design((past[:,::-1],hold,cp[2]),dt[::-1],t,tau)[:,:,:2]@th[:2]))]:
  for k,b in [('linear',lin[:,None]),('unit',e(1.)),('branch',exp),('all',np.c_[lin,e(1.),exp])]:
   A=drift[:,None,None]*b; r=gy-base; a0=np.linalg.lstsq((A/ws[None,:,None]).reshape(-1,b.shape[1]),(r/ws).ravel(),rcond=None)[0]; from scipy.optimize import least_squares
   coef=least_squares(lambda q:((A@q-r)/ws).ravel(),a0,loss='soft_l1').x; err=A@coef-r; print(mode,k,np.round(coef,4),np.round([np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])),np.quantile(abs(err),.9)],4));
 for k in ['01','02','03']:
  path=p.replace('01_solver',k+'_solver'); spec=importlib.util.spec_from_file_location('arc'+k,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  if not hasattr(mod,'_chamber_predict'): pr= np.c_[a@th]
  elif k=='01': pr=np.column_stack(mod._chamber_predict(cp,t,tau,th))
  else: pr=np.column_stack(mod._chamber_predict(cp,dt,t,tau,th))
  er=pr-gy; print('archived',k,np.round([np.median(abs(er[:,:-1])),np.median(abs(er[:,-1])),np.quantile(abs(er),.9)],4))
