import importlib.util, pickle, numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive')
def load(name):
 s=importlib.util.spec_from_file_location(name,p/f'alpha_m98_v2_c01_{name}_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('batch_05')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=i['dataset']; tr,te,g=d['train'],d['test'],i['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; k=len(t)
bridge=m._axial_load_map(d['calibration_load_cells']); xt,xe=m._features(tr,te); sp=m._specimen(tr,xt,bridge)
tau,theta=m._fit(sp,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,theta)
def observed(g):return np.column_stack((g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']))
def joined(groups):return {key:np.concatenate([group[key] for group in groups]) for key in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
Y=observed(g)
print('diag output ranges',np.round(np.quantile(Y,[0,.5,1],axis=0),3)); print('lab sizes',np.unique(g['lab_id'],return_counts=True));print('diag missing',np.isnan(g['coupon_features_raw']).sum(axis=0))
def report(label,P):
 E=P-Y; print(label,'curve/rate rmse',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),'median abs',np.median(np.abs(E[:,:-1])),np.median(np.abs(E[:,-1])), 'worst lab median',max(np.median(np.abs(E[g['lab_id']==l,:-1])) for l in np.unique(g['lab_id'])))
R=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]; S=np.r_[-np.expm1(-t),1.];
for norm,x in [('independent',m._features(g,g)[0]),('joint',m._features(joined([tr,te]),g)[1])]:
 past,hold,F=m._specimen(g,x,bridge); drift=F@theta[2:]; C,B=m._branches((past,hold,F),dt,t,tau); bench=np.column_stack((C@theta[:2],B@theta[:2])); reset=(hold[:,None]*theta[:2])@R.T
 for label,P in [('bench-linear',bench+drift[:,None]*T),('bench-sat',bench+drift[:,None]*S),('reset-linear',reset+drift[:,None]*T),('reset-sat',reset+drift[:,None]*S),('reset-branch',reset+(drift[:,None]*[.34273982,2.13827756])@R.T)]:report(norm+' '+label,P)
 for name in ('batch_02','batch_03'):
  mod=load(name)
  if hasattr(mod,'_chamber_predict'):
   C,B=mod._chamber_predict((past,hold,F),dt,t,tau,theta);report(norm+' '+name,np.column_stack((C,B)))
 for label,base in [('bench',bench),('reset',reset)]:
  E=Y-base; A=np.stack([drift[:,None]*v for v in (R[:,0],R[:,1],T)],axis=-1); coef=np.linalg.lstsq(A.reshape(-1,3),E.ravel(),rcond=None)[0]; report(norm+' '+label+' fit drift',base+A@coef);print('drift coefficients',coef)
  H=np.linalg.lstsq(np.column_stack((hold,past,drift)),E,rcond=None)[0];print('remaining coefficients by input, horizons [0,4,last,rate]:',np.round(H[:,[0,4,-2,-1]],4).tolist())
