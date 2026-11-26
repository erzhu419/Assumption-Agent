import importlib.util, pickle, types
from pathlib import Path
import numpy as np
p=Path('/holdout_task/alpha_w02_batch_02_solver.py')
spec=importlib.util.spec_from_file_location('baseline',p); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=p.read_text().replace('range(6):','range(9):').replace('(1+(r/2.5)**2)**(-0.25)','(1+(r/2.5)**2)**(-0.5)').replace("loss='soft_l1'", "loss='cauchy'")
c=types.ModuleType('cauchy'); exec(compile(s,str(p),'exec'),c.__dict__)
e=pickle.load(open('/holdout_task/public_alpha_w02_batch_02.pkl','rb'))[0]; d=e['dataset']; g=e['gt']; t=d['train']; u=d['test']; n=len(t['lab_id']); tm=d['eval_times']; dt=d['past_dt']; a=b._inputs(d); tr=tuple(x[:n] for x in a); te=tuple(x[n:] for x in a)
y=np.c_[t['strain_curves_obs']-t['strain_0'][:,None],t['instant_rate_obs']]; missing=np.isnan(t['coupon_features_raw']).any(axis=1); missing_te=np.isnan(u['coupon_features_raw']).any(axis=1)
for model,complete in [(b,False),(c,False),(c,True)]:
 print('\nMODEL',model.__name__,complete)
 residual=np.empty_like(y)
 for lab in range(4):
  val=t['lab_id']==lab; fit=~val & (~missing if complete else True); tau,coef=model._fit(tuple(v[fit] for v in tr),dt,tm,y[fit]); residual[val]=model._design(tuple(v[val] for v in tr),dt,tm,tau,0)@coef-y[val]
 for group in (False,True):
  for alpha in (0.,.25,.5,.75,1.):
   rates=residual[:,-1]; r=residual[:,:-1]-alpha*rates[:,None]*tm
   print('OOF missing',group,'alpha',alpha,'worst lab median/90p curve',np.max([np.quantile(abs(r[(missing==group)&(t['lab_id']==lab)]),[.5,.9]) for lab in range(4)],axis=0).round(5),'worst endpoint median',max(np.median(abs(r[(missing==group)&(t['lab_id']==lab),-1])) for lab in range(4)).round(5))
 fit=~missing if complete else np.ones(n,dtype=bool); tau,coef=model._fit(tuple(v[fit] for v in tr),dt,tm,y[fit]); delta,z=model._components(te,dt,tau); amp=delta*coef[:2]; drift=z@coef[2:]; prior_rate=amp@(1/tau)+drift; dr=u['instant_rate_obs']-prior_rate
 print('taus',tau,'coefficients',coef,'test missing rows [index,rate innovation]',np.c_[np.where(missing_te)[0],dr[missing_te]].round(5))
 for alpha in (0.,.5,.75,1.):
  slope=drift+alpha*missing_te*dr
  pred=u['strain_0'][:,None]+amp@(-np.expm1(-tm[:,None]/tau)).T+slope[:,None]*tm
  sec=(pred[:,-1]-u['strain_0'])/np.maximum(abs(te[1]),.5); hz=model._hazard(u['strain_0'],u['critical_strain'],tm,tau,amp,slope)
  errors=[pred-g['te_clean'],sec-g['te_secant'],hz-g['te_cross']]
  print('public alpha',alpha,'RMSE',np.array([np.sqrt(np.mean(q*q)) for q in errors]).round(6),'max',np.array([np.max(abs(q)) for q in errors]).round(6),'hazard censor disagreements',np.sum((hz==tm[-1])!=(g['te_cross']==tm[-1])))
