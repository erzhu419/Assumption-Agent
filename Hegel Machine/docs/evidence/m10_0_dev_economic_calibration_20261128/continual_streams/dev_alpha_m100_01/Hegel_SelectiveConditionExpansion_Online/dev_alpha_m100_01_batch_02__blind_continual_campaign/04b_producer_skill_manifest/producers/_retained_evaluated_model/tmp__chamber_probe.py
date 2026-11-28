import sys, pickle, json, importlib.util
import numpy as np
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_alpha_m100_01_batch_01_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=p['dataset']; diag=p['campaign_diagnostics']; d=calibrate_upstream(raw); q=calibrate_upstream({**raw,'test':diag})['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; n=len(tr['lab_id']); xt=old._features(tr,te); loads=lambda z: tuple(z[k].sum(axis=-1) for k in ('past_channel_raw','hold_channel_raw')); y=lambda z:z['strain_curves_obs']-z['strain_0'][:,None]; taus,th=old._fit(*loads(tr),xt[:n],dt,t,y(tr),tr['instant_rate_obs']); hist,hold=loads(q); obs=np.column_stack((y(q),q['instant_rate_obs']));
def baseline():
 c,r=old._design(taus,*loads(tr),xt[:n],dt,t); print('taus, theta',taus,th); e=np.column_stack((c@th-y(tr),r@th-tr['instant_rate_obs'])); print('train med/90% abs strain/rate',np.quantile(abs(e[:,:-1]),[.5,.9]),np.quantile(abs(e[:,-1]),[.5,.9]));
if __name__=='__main__':
 baseline();
 for label,x in [('diag only',old._features(q,q)[:len(hold)]),('train+diag',old._features(tr,q)[n:]),('all+diag',old._features({k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')},q)[-len(hold):])]:
  sc=th[2]+x@th[3:]; c,r=old._design(taus,hist,hold,x,dt,t); prev,pr=old._chamber(taus,th,hold,x,t); print(label,'base/prev median abs',*[np.median(abs(e),axis=0).round(5).tolist() for e in (np.column_stack((c@th,r@th))-obs,np.column_stack((prev,pr))-obs)]);
  if label!='train+diag':continue
  K=np.vstack((np.column_stack((-np.expm1(-t[:,None]/taus),t)),np.r_[1/taus,1.])); coef=np.linalg.lstsq(K,obs.T,rcond=None)[0].T; err=obs-coef@K.T; print('free canonical kernel residual med/90%',np.median(abs(err),axis=0).round(6),np.quantile(abs(err),.9,axis=0).round(6)); print('free coefs, hold and score',np.c_[coef,hold,sc].round(5)); dsg=np.column_stack((hold,hist,sc,x,np.ones(len(hold)))); print('correlations (hold,past0..5,score,X0..3, then coefs)',np.corrcoef(np.column_stack((dsg[:,:-1],coef)).T)[-3:,:-3].round(3));
  for idx in ([0,7],[0,7,12],[0,7,1,2,3,4,5,6,12],[0,8,9,10,11,12]):
   A=dsg[:,idx]; b=np.linalg.lstsq(A,coef,rcond=None)[0]; print(idx,'weights',b.round(5),'curve/rate residual quantiles',np.quantile(abs((coef-A@b)@K.T),[.5,.9],axis=0).round(5))
