import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'
s=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
for item in items:
 d=item['dataset']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=m._features(tr,te)[:n]; c=d['calibration_load_cells']; B=m._fit_bridge(c)
 past,hold=m._loads(tr,B); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; labs=tr['lab_id']; pred=m.solve_batch(d); tau=pred['estimated_taus']; theta=pred['hereditary_weights']; print('all-coupon fit',tau,theta)
 assert .05<tau[0]<tau[1]<25 and all(np.isfinite(v).all() for v in pred.values())
 ec=np.empty_like(yc); er=np.empty_like(yr)
 for lab in np.unique(labs):
  u=labs!=lab; v=~u
  q,b=m._fit(x[u],past[u],hold[u],dt,t,yc[u],yr[u]); D,R=m._design(x[v],past[v],hold[v],dt,t,q)
  ec[v]=(D@b).reshape(-1,len(t))-yc[v]; er[v]=R@b-yr[v]
  print('held lab',lab,'taus',q,'max weights change',np.max(np.abs(theta-b)),'curve/rate MAE',np.mean(abs(ec[v])),np.mean(abs(er[v])),'medAE',np.median(abs(ec[v])),np.median(abs(er[v])))
 for idx in [0,len(t)-1]: print('horizon',t[idx],'lab MAE',[np.mean(abs(ec[labs==l,idx])) for l in np.unique(labs)])
 pp,hh=m._loads(te,B); z=np.zeros((len(te['lab_id']),2));
 for j,step in enumerate(dt):
  a=np.exp(-step/tau); z=a*z+(1-a)*pp[:,j,None]
 gap=hh[:,None]-z; drift=theta[2]+m._features(tr,te)[n:]@theta[3:]
 curve=te['strain_0'][:,None]+(gap*theta[:2])@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*t
 rate=(gap*theta[:2]/tau).sum(1)+drift
 np.testing.assert_allclose(pred['predicted_strain'],curve,atol=1e-10)
 np.testing.assert_allclose(pred['initial_rate_pred'],rate,atol=1e-10)
 print('test curve and initial rate agree with independent chronological recurrence')
