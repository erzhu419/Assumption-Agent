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
A=np.empty((len(P),0)); print('diag keys',list(q)); print('memory singular values',np.linalg.svd(Zc-Z,compute_uv=False));
V={f'aux{i}':A[:,i] for i in range(A.shape[1])}; V.update({f'x{i}':x[:,i] for i in range(x.shape[1])}); V.update({f'past{i}':P[:,i] for i in range(P.shape[1])}); V.update({f'diff{i}':u for i,u in enumerate(np.diff(P,axis=1).T)}); V.update(hold=H,z0=Z[:,0],z1=Z[:,1],r=r,loadmean=P.mean(1),loadstd=P.std(1),loadmin=P.min(1),loadmax=P.max(1),negmean=np.minimum(P,0).mean(1),posmean=np.maximum(P,0).mean(1),path=np.abs(np.diff(P,axis=1)).sum(1));
v=list(V); B=V.copy();
for i,k in enumerate(v):
 u=V[k]; B[k+'^2']=u*u; B[k+'+']=u.clip(0); B[k+'-']=(-u).clip(0)
 for l in v[:i]:
  w=V[l]; B[k+'*'+l]=u*w; B[k+'+'+l]=u+w; B[k+'-'+l]=u-w; B['abs('+k+'-'+l+')']=abs(u-w)
for target in [yr,(Zc-Z)[:,0],(Zc-Z)[:,1]]:
 def best(u):
  if np.std(u)<1e-9:return float('inf')
  X=np.column_stack((np.ones(len(u)),u)); I=np.linalg.pinv(X); e=target-X@(I@target); return np.mean((e/np.maximum(1-np.sum(X*I.T,axis=1),1e-8))**2)**.5
 print('linear LOO screen', sorted((round(best(u),5),k) for k,u in B.items())[:25]);
