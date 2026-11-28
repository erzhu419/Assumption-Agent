import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m100_c02_batch_03_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,v=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']
print('sizes',len(tr['lab_id']),len(te['lab_id']),len(v['lab_id']),'cal fields',[(k,type(a).__name__,getattr(a,'shape',None)) for k,a in cal.items()]); print('labs/strata',[(np.unique(a['lab_id'],return_counts=True),np.unique(a['stratum'],return_counts=True)) for a in (tr,te,v)])
c=m._fit_bridge(cal); err=np.sum(m._bridge_basis(cal['true_channel_loads'])*c,axis=-1)-cal['bridge_raw']; print('bridge abs residual q',np.quantile(abs(err),[.5,.9,.99,1],axis=0)); print('coef',c)
pt,ht=m._loads(c,tr); pv,hv=m._loads(c,v); x=m._features(tr,te); xv=m._features(dict((k,np.concatenate([tr[k],te[k]])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')),v)[-len(hv):]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=m._fit(pt,ht,x[:len(ht)],dt,t,y,r); print('material',tau,theta)
for tag,s,h,xx,Y,R in [('train',pt,ht,x[:len(ht)],y,r),('diag',pv,hv,xv,v['strain_curves_obs']-v['strain_0'][:,None],v['instant_rate_obs'])]:
 a,b,*_=m._design(np.log(tau),s,h,xx,dt,t); pred,rate=a@theta,b@theta; old,oldrate=m._deployment(tau,theta,xx,s,h,t)
 print(tag,'ranges',np.quantile(s,[0,.5,1]),np.quantile(h,[0,.5,1])); print('abs residual median/p90/max',[(np.quantile(abs(u-Y),[.5,.9,1]).round(6), np.quantile(abs(w-R),[.5,.9,1]).round(6)) for u,w in ((pred,rate),(old,oldrate))]);
 if tag=='diag':
  mat=np.column_stack([-np.expm1(-t[:,None]/tau),t]); rate_basis=np.r_[1/tau,1]; print('basis cond',np.linalg.cond(mat)); amp=np.linalg.lstsq(np.vstack([mat,rate_basis]),np.column_stack([Y,R]).T,rcond=None)[0].T; print('raw amplitude fit abs residual',np.quantile(abs(amp@mat.T-Y),[.5,.9,1])); expected=np.column_stack([b[:,:2]*tau*theta[:2],xx@theta[2:]]); print('columns: diag lab/stratum, hold, baseline amplitudes, fitted amplitude corrections, baseline rate correction'); print(np.column_stack([v['lab_id'],v['stratum'],h,expected,amp-expected,R-rate]).round(5));
