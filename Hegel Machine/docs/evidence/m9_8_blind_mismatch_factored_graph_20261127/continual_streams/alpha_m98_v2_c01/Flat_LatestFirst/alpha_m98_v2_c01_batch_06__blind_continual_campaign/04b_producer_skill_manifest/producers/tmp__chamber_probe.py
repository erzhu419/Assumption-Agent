import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m98_v2_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; cal=d['calibration_load_cells']; bridge=m._axial_load_map(cal)
a=cal['true_channel_loads']; b=cal['bridge_raw']; M=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0]; print('calibration forward RMSE',np.sqrt(np.mean((np.c_[a,np.ones(len(a))]@M-b)**2,axis=0)),'axial RMSE',np.sqrt(np.mean(((b-bridge[0])@bridge[1]-a.sum(1))**2)));print('bridge matrix',M.round(4))
xt,xe=m._features(tr,te); tau,th=m._fit(m._specimen(tr,xt,bridge),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau/theta',tau,th)
y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; E=-np.expm1(-t[:,None]/tau); E1=-np.expm1(-t)
def stats(a):
 return np.round([np.median(abs(a)),np.sqrt(np.mean(a*a)),np.max(abs(a))],6)
for label,ref in [('g',g),('tr',tr),('te',te),('tr+te',{k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})]:
 if label=='g': _,x=m._features(g,g)
 else: _,x=m._features(ref,g)
 p,h,X=m._specimen(g,x,bridge); c,rr=m._branches((p,h,X),dt,t,tau); dr=X@th[2:]; B=c@th[:2]; R=rr@th[:2]; print(label, 'base/exp curves',stats(y-B-dr[:,None]*t),stats(y-B-dr[:,None]*E1),'rate',stats(r-R-dr));
 if label=='tr+te':
  np.set_printoptions(precision=5,suppress=True,linewidth=145);print('rate: obs, branch, drift, residual');print(np.c_[r,R,dr,r-R-dr]);print('curves residual med by time linear',np.median(y-B-dr[:,None]*t,axis=0)); print('exp',np.median(y-B-dr[:,None]*E1,axis=0));
  print('past,hold ranges',p.min(),p.max(),h.min(),h.max());
  for name,F,df in [('linear',t,1),('exp1',E1,1),('zero',t*0,0)]:
   A=np.vstack((E,1/tau)); obs=np.c_[y-dr[:,None]*F,r-dr*df]; coeff=np.linalg.lstsq(A,obs.T,rcond=None)[0].T; print(name,'curve+rate residual',stats(obs-coeff@A.T),'amplitudes actual vs baseline (first 8)',np.c_[coeff,c[:,0,:]/E[0]*th[:2]][:8]);
