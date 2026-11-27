import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('s','/holdout_task/alpha_m96_w04_batch_01_solver.py'); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_alpha_m96_w04_batch_01.pkl','rb'))[0]['dataset']; a,b=d['train'],d['test']; n=len(a['lab_id']); x=s._align(a,b); m=s._Creep(a,x[:n],s._calibrate(d['calibration_load_cells']),d['past_dt'],d['eval_times']); y=np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs'])); miss=~np.isfinite(a['coupon_features_raw']).all(axis=1); t=d['eval_times']; labs=a['lab_id']; names=['baseline','rate_missing','rate_all','complete_fit_missing_rate']; r={k:np.empty((n,len(t))) for k in names}
for lab in np.unique(labs):
 h=labs==lab
 for i,fitmask in enumerate((~h,~h & ~miss)):
  p=s._fit(m,y,fitmask); v=(m.basis(p[:2])[0]@p[2:])[h]; base=v[:,:-1]-y[h,:-1]; slope=v[:,-1]-y[h,-1]
  if i==0:
   r['baseline'][h]=base
   r['rate_missing'][h]=base-slope[:,None]*t*miss[h,None]
   r['rate_all'][h]=base-slope[:,None]*t
  else:
   r['complete_fit_missing_rate'][h]=base-slope[:,None]*t*miss[h,None]
for name,q in r.items():
 print(name)
 for lab in np.unique(labs):
  for missing in (False,True):
   sel=(labs==lab)&(miss==missing);
   if not np.any(sel): continue
   z=q[sel]; print(int(lab),missing,len(z),np.round([np.sqrt(np.mean(z**2)),np.sqrt(np.mean(z[:,-1]**2)),np.median(np.median(abs(z),axis=1)),np.max(np.median(abs(z),axis=1)),np.median(abs(z[:,-1]))],5))
