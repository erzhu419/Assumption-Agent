import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w04_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
w=pickle.load(open('/holdout_task/public_alpha_w04_batch_01.pkl','rb'))[0]; d=w['dataset']; a,b=d['train'],d['test']; q=w['gt']; t=d['eval_times']; dt=d['past_dt']; cal=m._calibrate(d['calibration_load_cells']); x,xt=m._align(a,b); h,hold=m._stress(a,cal); ht,st=m._stress(b,cal); y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; z=a['initial_memory_states']; lab=a['lab_id']; wing=lab//2; ok=np.isfinite(a['coupon_features_raw']).all(1)
shares=[0.,.5,1.]; errors={k:np.zeros_like(y) for k in shares}
for l in [0,1,2,3,None]:
 tr=ok if l is None else (lab!=l)&ok
 tau,beta=m._fit(dt,t,h[tr],hold[tr],x[tr],y[tr],r[tr],z[tr]); v,u=m._design(tau,dt,t,h[tr],hold[tr],x[tr],z[tr]); f=beta[2]+x[tr]@beta[3:]; rel=v[:,:,:2]@beta[:2]; residual=np.median((y[tr,5:]-rel[:,5:])/t[5:],axis=1)-f
 offs=np.array([np.median(residual[wing[tr]==i]) for i in range(2)]); print('held-out lab',l,'wing drift offsets',offs)
 if l is None: te=b; xx=xt; hh=ht; ss=st
 else:
  mask=lab==l; te={k:value[mask] for k,value in a.items()}; xx=x[mask]; hh=h[mask]; ss=hold[mask]
 v,u=m._design(tau,dt,t,hh,ss,xx,te['initial_memory_states']); f=beta[2]+xx@beta[3:]; measured=te['instant_rate_obs']-u[:,:2]@beta[:2]; ok_te=np.isfinite(te['coupon_features_raw']).all(1); rel=v[:,:,:2]@beta[:2]; start=te['strain_0']
 for k in shares:
  feat=f+k*offs[te['lab_id']//2]; drift=np.where(ok_te,feat+.5*np.clip(measured-feat,-.035,.035),measured); inc=rel+drift[:,None]*t
  if l is not None: errors[k][mask]=inc-y[mask]; continue
  p=start[:,None]+inc; sec=inc[:,-1]/np.maximum(np.abs(ss),.5); cross=m._crossing(t,start,p,te['critical_strain']); es=[p-q['te_clean'],sec-q['te_secant'],cross-q['te_cross']]; print('public correction',k,'worst lab RMSE',[round(max(np.sqrt(np.mean(e[te['lab_id']==j]**2)) for j in range(4)),6) for e in es],'pools',[round(np.sqrt(np.mean(e**2)),6) for e in es])
for k,e in errors.items():
 print('LOLO correction',k,'worst complete lab terminal median',max(np.median(abs(e[(lab==j)&ok,-1])) for j in range(4)),'worst clipped curve RMSE',max(np.sqrt(np.mean(np.minimum(abs(e[(lab==j)&ok]),.075)**2)) for j in range(4)))
