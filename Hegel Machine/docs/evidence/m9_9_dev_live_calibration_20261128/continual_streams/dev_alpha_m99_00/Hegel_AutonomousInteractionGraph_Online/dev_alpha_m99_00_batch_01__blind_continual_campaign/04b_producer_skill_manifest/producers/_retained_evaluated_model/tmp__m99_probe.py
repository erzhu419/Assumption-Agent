import pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m99_00_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=it['dataset']; tr,te,di=ds['train'],ds['test'],it['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']; n=len(tr['lab_id']); bridge=m._bridge(ds['calibration_load_cells']); x=m._features(tr,te,di); x0=m._features(tr,te)
def obs(d): return d['strain_curves_obs']-d['strain_0'][:,None], d['instant_rate_obs']
ct=m._coupon(tr,x0[:n],bridge); cd=m._coupon(di,x[-len(di['lab_id']):],bridge); tau,th=m._fit(ct,dt,t,*obs(tr)); print('material',tau,th,'dt',dt,'times',t);print('feature panel sensitivity',np.sqrt(np.mean((x[:len(x0)]-x0)**2)))
cal=ds['calibration_load_cells']; B=np.c_[cal['bridge_raw'],np.ones(len(cal['bridge_raw']))]; e=B@bridge-cal['true_channel_loads'].sum(axis=1);print('calibration rms/max',np.sqrt(np.mean(e**2)),np.abs(e).max())
def summary(label,r,q): print(label,'curve RMS/median/max',np.sqrt(np.mean(r*r)),np.median(abs(r)),np.max(abs(r)),'rate RMS/median/max',np.sqrt(np.mean(q*q)),np.median(abs(q)),np.max(abs(q)))
for d,c in [(tr,ct),(di,cd)]:
 y,dy=obs(d); A,R=m._design(tau,c,dt,t); summary('baseline',A@th-y,R@th-dy)
 def mat(z):
  u=np.exp(z);return np.r_[np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1][None,:]]
 Y=np.c_[y,dy];
 def fit(z):
  a=mat(z);return (np.linalg.lstsq(a,Y.T,rcond=None)[0].T@a.T-Y).ravel()
 for u in [tau,[.2,2],[.7,5]]:
  sol=least_squares(fit,np.log(u),bounds=(np.log(.051),np.log(24.99)));print('free amplitude times',np.exp(sol.x),np.sqrt(np.mean(sol.fun**2)))
 Z=np.linalg.lstsq(mat(sol.x),Y.T,rcond=None)[0].T; r=Z@mat(sol.x).T-Y;summary('free amplitude',r[:,:-1],r[:,-1]); print('amplitude std',Z.std(0),'fit mean',Z.mean(0)); print('lab means',[(l,np.mean((A@th-y)[d['lab_id']==l]**2)**.5) for l in np.unique(d['lab_id'])]);
 if d is di:
  print('per time median residual',np.median(A@th-y,axis=0)); print('rate median residual',np.median(R@th-dy)); print('observed diag first three',Y[:3].round(4))
for split in [tr,te]:
 a=split['aux_chamber_telemetry']; print('aux mean/std/min/max',a.mean(0),a.std(0),a.min(0),a.max(0))
