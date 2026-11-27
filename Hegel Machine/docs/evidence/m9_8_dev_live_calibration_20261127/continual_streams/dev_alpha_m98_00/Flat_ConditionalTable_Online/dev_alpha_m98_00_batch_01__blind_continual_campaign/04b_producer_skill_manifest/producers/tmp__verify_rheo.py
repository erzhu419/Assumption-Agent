import runpy,pickle,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; q=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; out=m['solve_batch'](d); tau=out['estimated_taus']; th=out['hereditary_weights']; c=m['_stress_calibration'](d['calibration_load_cells']); assert .05<tau[0]<tau[1]<25
hold=q['hold_channel_raw']@c[:-1]+c[-1]; x=m['_aligned_features'](q,q)[len(hold):]; drift=x@th[2:]; f=np.array([.52,.43]); J=-np.expm1(-t[:,None]/tau); amp=hold[:,None]*th[:2]+drift[:,None]*f*tau; ec=abs(q['strain_curves_obs']-q['strain_0'][:,None]-amp@J.T-(1-f.sum())*drift[:,None]*t); er=abs(q['instant_rate_obs']-amp@(1/tau)-(1-f.sum())*drift)
for name,labels in [('lab',q['lab_id']),('stratum',q['stratum'])]:
 for j in np.unique(labels):
  s=labels==j; print(name,j,'curve P90 and rate P90',np.quantile(ec[s],.9),np.quantile(er[s],.9))
train_coupon,test_coupon=m['_prepare'](d); _,h,X=test_coupon; np.testing.assert_allclose(out['initial_rate_pred'],h*(th[:2]/tau).sum()+X@th[2:],rtol=0,atol=1e-12); assert all(np.isfinite(a).all() for a in out.values())
A,R=m['_design'](tau,dt,t,train_coupon); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; v=tr['instant_rate_obs']; print('intrinsic',tau,th)
for label in ('lab_id','stratum'):
 for j in np.unique(tr[label]):
  k=tr[label]!=j; tt,ww=m['_fit'](dt,t,tuple(a[k] for a in train_coupon),y[k],v[k]); AA,RR=m['_design'](tt,dt,t,tuple(a[~k] for a in train_coupon)); print('held out',label,j,'taus',tt,'curve and rate P90',np.quantile(abs(AA@ww-y[~k]),.9),np.quantile(abs(RR@ww-v[~k]),.9))
