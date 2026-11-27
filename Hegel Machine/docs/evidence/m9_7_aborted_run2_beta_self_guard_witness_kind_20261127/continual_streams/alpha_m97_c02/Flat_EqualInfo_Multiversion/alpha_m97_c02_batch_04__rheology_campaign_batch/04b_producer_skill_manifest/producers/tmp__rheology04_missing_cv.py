import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibration(d); x,_=s._aligned(tr,d['test']); args=s._inputs(tr,x,cal,dt,d['crit_stress'],d['shear_banding_index']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; active=args[-1]; good=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for complete_only in (False,True):
 slices=[]
 for lab in range(4):
  held=tr['lab_id']==lab; fit=~held & (good if complete_only else True)
  tau,w=s._fit(dt,t,tuple(a[fit] for a in args),y[fit],r[fit]); a,b=s._design(tau,dt,t,*(a[held] for a in args)); ec=a@w-y[held]; er=b@w-r[held]; print('complete_only',complete_only,'lab',lab,'tau',np.round(tau,4))
  for mask in (active[held],~active[held],good[held],~good[held]):
   if mask.any(): slices.append([np.median(abs(ec[mask])),np.quantile(abs(ec[mask]),.9),np.sqrt(np.mean(er[mask]**2))])
 print('worst lab-phase and lab-missingness slice: median AE, p90 AE, rate RMSE',np.max(slices,axis=0))
 tau,w=s._fit(dt,t,tuple(a[good] if complete_only else a for a in args),y[good] if complete_only else y,r[good] if complete_only else r); print('full training tau',tau)
