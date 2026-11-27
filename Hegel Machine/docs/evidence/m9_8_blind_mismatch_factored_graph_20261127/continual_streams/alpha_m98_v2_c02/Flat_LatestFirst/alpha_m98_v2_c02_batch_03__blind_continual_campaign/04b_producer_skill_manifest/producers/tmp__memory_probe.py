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
from scipy.optimize import minimize_scalar
P=stress(q['past_channel_raw']); H=stress(q['hold_channel_raw']); tail=np.cumsum(dt[::-1])[::-1]-dt
W=lambda tt: -np.expm1(-dt[:,None]/np.atleast_1d(tt))*np.exp(-tail[:,None]/np.atleast_1d(tt))
r=beta[2]+x@beta[3:]
G=lambda a:T+a*np.expm1(-T/a)
E=lambda a:(Y+r[:,None]*G(a))
Q=np.eye(len(T))-F@np.linalg.pinv(F)
a=minimize_scalar(lambda a:np.sum((E(a)@Q)**2),bounds=(.1,4),method='bounded').x
C=E(a)@np.linalg.pinv(F).T
Z=P@W(tau); Zc=Z-C/beta[:2]
print('drift ramp duration',a,'orthogonal RMSE',np.mean((E(a)@Q)**2)**.5)
print('row, baseline memory, inferred chamber memory, delta, base drift and rate correction');print(np.column_stack((np.arange(len(P)),Z,Zc,Zc-Z,r,yr)).round(4))
# Search for a single altered decay clock, affine calibration, and common nonlinear drive maps.
trans={'raw':P,'square':P*P,'positive':P.clip(0),'tanh':np.tanh(P),'offset_square':(P-H[:,None])**2,'gap_positive':(H[:,None]-P).clip(0), 'last':np.broadcast_to(P[:,-1:],P.shape), 'hold':np.broadcast_to(H[:,None],P.shape)}
for j in (0,1):
 for name,A in trans.items():
  for form in ('replace','add'):
   def mat(tt):
    u=(A@W(tt)).ravel(); return np.column_stack((np.ones(len(P)),u)) if form=='replace' else np.column_stack((np.ones(len(P)),Z[:,j],u))
   def fit(tt):
    X=mat(tt); inv=np.linalg.pinv(X); e=Zc[:,j]-X@inv@Zc[:,j]; loo=e/np.maximum(1-np.sum(X*inv.T,axis=1),1e-8); return np.mean(loo**2)**.5
   res=minimize_scalar(fit,bounds=(.06,12),method='bounded'); X=mat(res.x); c=np.linalg.lstsq(X,Zc[:,j],rcond=None)[0]
   print(j,name,form,'tau',round(res.x,3),'loo',round(res.fun,4),'coeff',c.round(3))
print('aux',[(g['aux_chamber_telemetry'].shape,np.asarray(g['aux_chamber_telemetry'])[:2].round(4).tolist()) for g in (tr,te)])
