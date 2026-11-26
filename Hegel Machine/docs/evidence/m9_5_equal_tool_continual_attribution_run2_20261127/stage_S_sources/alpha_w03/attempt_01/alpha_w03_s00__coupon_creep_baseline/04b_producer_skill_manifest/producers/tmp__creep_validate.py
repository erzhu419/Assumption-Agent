import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w03_s00.pkl','rb'))[0]['dataset']; a=d['train']; x,h,p=m._prep(d); dt,t=d['past_dt'],d['eval_times']; y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; n=len(y); x,h,p=x[:n],h[:n],p[:n]
for col in ['lab_id','stratum']:
 for i in np.unique(a[col]):
  v=a[col]==i; f=~v; tau,w=m._fit(x[f],h[f],p[f],dt,t,y[f],r[f]); c,b,*_=m._design(tau,x[v],h[v],p[v],dt,t); ce=np.abs(c@w-y[v]); re=np.abs(b@w-r[v]); print(col,i,'taus',tau.round(5),'curve med/p90/worst horizon med',np.r_[np.quantile(ce,[.5,.9]),np.max(np.median(ce,axis=0))].round(6),'rate med/p90',np.quantile(re,[.5,.9]).round(6))
  if col=='lab_id':
   missing=np.isnan(a['coupon_features_raw'][v]).any(axis=1)
   if missing.any(): print('missing features',missing.sum(),'curve/rate med',np.median(ce[missing]),np.median(re[missing]))
o=m.solve_alpha_w03_s00(d); print('full taus and weights:',o['estimated_taus'],o['hereditary_weights']); tau=o['estimated_taus']; z=np.zeros((n,2));
for j,step in enumerate(dt):
 e=np.exp(-step/tau); z=e*z+(1-e)*p[:,j,None]
c,b,*_=m._design(tau,x,h,p,dt,t); assert np.max(np.abs(b[:,:2]-(h[:,None]-z)/tau))<1e-12
print('exact recurrence agreement')
c=d['calibration_load_cells'];
for i in range(4):
 v=c['lab_id']==i; u=c['cmd_stress'][v]; q=c['true_stress'][v]; w=m._line(u,q); j=np.argmax(np.abs(u)); q=q.copy(); q[j]+=.9; w2=m._line(u,q); print('EMI at extreme command',i,'gain/bias shift',np.abs(w-w2).round(6))
