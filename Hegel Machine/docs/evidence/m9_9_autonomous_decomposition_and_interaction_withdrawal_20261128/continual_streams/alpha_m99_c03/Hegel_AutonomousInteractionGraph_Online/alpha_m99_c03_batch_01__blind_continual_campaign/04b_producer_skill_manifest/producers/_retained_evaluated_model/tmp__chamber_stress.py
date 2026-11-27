import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; q=it['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; n=len(tr['lab_id']); x=m._features(tr,te); b=m._calibration(d['calibration_load_cells']); ai=m._coupon_inputs(tr,x[:n],b); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; full=m.solve_batch(d)
qx=m._features(q,{k:q[k][:0] for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}); _,h,dx=m._coupon_inputs(q,qx,b); complete=np.isfinite(q['coupon_features_raw']).all(1); target=q['strain_curves_obs']-q['strain_0'][:,None]; rate=q['instant_rate_obs']
for lab in [None,*np.unique(tr['lab_id'])]:
 mask=np.ones(n,bool) if lab is None else tr['lab_id']!=lab
 a,w=(full['estimated_taus'],full['hereditary_weights']) if lab is None else m._fit(tuple(z[mask] for z in ai),dt,t,yc[mask],yr[mask]); f=-np.expm1(-t[:,None]/a); r=dx@w[2:]; er=h*np.sum(w[:2]/a)+r-rate; base=h[:,None]*(f@w[:2]);
 print('omitted train lab',lab,'taus',np.round(a,5),'rate RMSE complete/missing',*[round(np.sqrt(np.mean(er[j]**2)),5) for j in (complete,~complete)])
 for shares in ([.5,.45],[.5,.475],[.5,.5]):
  k=f@(a*shares)+(1-sum(shares))*t; e=base+r[:,None]*k-target
  print(shares,'RMSE complete/missing',*[round(np.sqrt(np.mean(e[j]**2)),5) for j in (complete,~complete)],'max',round(abs(e).max(),5),'worst lab RMSE',round(max(np.sqrt(np.mean(e[q['lab_id']==i]**2)) for i in np.unique(q['lab_id'])),5))
