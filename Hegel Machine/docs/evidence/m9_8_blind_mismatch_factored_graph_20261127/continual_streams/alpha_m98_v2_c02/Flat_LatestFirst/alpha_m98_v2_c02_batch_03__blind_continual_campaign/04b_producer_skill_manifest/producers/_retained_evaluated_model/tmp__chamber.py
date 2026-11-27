import glob, importlib.util, json, pickle
import numpy as np
p=glob.glob('/tmp/hegel_continual_archive/*batch_02_solver.py')[0]
s=importlib.util.spec_from_file_location('base',p); b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']
xtr,xte=b._standardize(tr,te); stress=b._stress_calibration(cal)
a=b._design(tr,xtr,stress,dt,t); tau,beta=b._fit(a,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('dt',dt,'times',t,'tau',tau,'beta',beta)
from scipy.optimize import least_squares
from scipy.special import expit
x=b._standardize({k:np.concatenate((tr[k],te[k])) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']},q)[1]
S,R=b._design(q,x,stress,dt,t)(tau)
y=q['strain_curves_obs']-q['strain_0'][:,None]-S@beta; yr=q['instant_rate_obs']-R@beta
f=-np.expm1(-t[:,None]/tau); Y=np.column_stack((y,yr)); T=np.r_[t,0.]; F=np.column_stack((f.T,1/tau)).T
funcs={'integrated_exp':lambda z: T+np.expm1(-T/z[0])*z[0], 't_exp':lambda z: T*-np.expm1(-T/z[0]),'smooth_hinge':lambda z: z[1]*(np.logaddexp(0,(T-z[0])/z[1])-np.logaddexp(0,-z[0]/z[1])), 'hinge':lambda z: np.maximum(0,T-z[0]), 'hill':lambda z:T*(T**z[1]/(T**z[1]+z[0]**z[1]))}
for name,fun in funcs.items():
 for branches in (1,2):
  n=2 if name in ('smooth_hinge','hill') else 1
  def basis(z): return np.column_stack((F[:,:branches], fun(z)))
  def resid(z):
   D=basis(z); return (Y-Y@np.linalg.pinv(D).T@D.T).ravel()
  z=least_squares(resid,[.6,.2][:n],bounds=(.01,[5,12][:n])).x; D=basis(z); C=Y@np.linalg.pinv(D).T; E=Y-C@D.T
  print(name,branches,z.round(4),'rmse',np.sqrt(np.mean(E**2)).round(5),'rate',np.sqrt(np.mean(E[:,-1]**2)).round(5));
  if branches==1: print(' coeffs',C.round(3).T)
  if name=='integrated_exp' and branches==2: C0=C
A=np.column_stack((q['lab_id'],q['stratum'],x,stress(q['hold_channel_raw']),stress(q['past_channel_raw']))); print('lab str x0:4 hold past',A.round(3)); print('corr',np.corrcoef(np.column_stack((A,C0)).T)[-3:,:-3].round(2)); print('ref lab med mean and std',[(j,*[tuple(np.quantile(v[g['lab_id']==j],p,axis=0).round(2) for p in [.5]) for g,v in [(q,q['ref_coupons'])]]) for j in np.unique(q['lab_id'])]); print('te ranges',np.quantile(xte,[0,.5,1],axis=0).round(3))
