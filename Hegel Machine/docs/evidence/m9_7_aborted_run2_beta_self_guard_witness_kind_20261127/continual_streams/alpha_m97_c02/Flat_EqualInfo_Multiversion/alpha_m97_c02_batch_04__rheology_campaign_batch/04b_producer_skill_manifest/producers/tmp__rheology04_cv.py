import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibration(d); x,_=s._aligned(tr,d['test']); args=s._inputs(tr,x,cal,dt,d['crit_stress'],d['shear_banding_index']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; active=args[-1]; design=s._design
for mode in ('gated','all_log','all_linear'):
 def alt(tau,dt,t,past,hold,flow,activation,active):
  a,b=design(tau,dt,t,past,hold,flow,activation,active)
  if mode=='all_log': a[:,:,2:7]=flow[:,None,:]*np.log1p(t)[None,:,None]
  if mode=='all_linear':
   a[:,:,2:7]=flow[:,None,:]*t[None,:,None]; a[:,:,8]=b[:,8,None]*t
  return a,b
 s._design=alt; ec=np.empty_like(y); er=np.empty_like(r)
 print(mode)
 for lab in range(4):
  test=tr['lab_id']==lab; fit=~test
  tau,w=s._fit(dt,t,tuple(a[fit] for a in args),y[fit],r[fit]); a,b=alt(tau,dt,t,*(a[test] for a in args)); ec[test]=a@w-y[test]; er[test]=b@w-r[test]
  for phase in (True,False):
   k=test&(active==phase)
   print(lab,int(phase),k.sum(),'curve MAE/p90',np.round([np.median(abs(ec[k])),np.quantile(abs(ec[k]),.9)],5),'rate RMSE',round(np.sqrt(np.mean(er[k]**2)),5),'tau',np.round(tau,5))
 print('horizon curve median AE',np.round(np.median(abs(ec),axis=0),5),'pooled rate RMSE',np.sqrt(np.mean(er**2)))
