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
G=lambda a:T+a*np.expm1(-T/a)
M=lambda a:np.column_stack((F,G(a)))
a=least_squares(lambda p:(Y-Y@np.linalg.pinv(M(p[0])).T@M(p[0]).T).ravel(),[1.2],bounds=(.1,5)).x[0]; D=M(a); C=Y@np.linalg.pinv(D).T
P=stress(q['past_channel_raw']); H=stress(q['hold_channel_raw']); tail=np.cumsum(dt[::-1])[::-1]-dt
W=-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau); Z=P@W
r=beta[2]+x@beta[3:]
print('row lab str miss corrections C0 C1 C2 / beta0,beta1 r and memory'); print(np.column_stack((np.arange(len(x)),q['lab_id'],q['stratum'],np.isnan(q['coupon_features_raw']).sum(1),C[:,:2]/beta[:2],C[:,2],r,Z,H,P[:,-1])).round(4))
print('G duration',a)
for variant,u in [('joint all',x),('diagnostic only',b._standardize(q,q)[1]),('train-test ref fixed',x.copy())]:
 if variant=='train-test ref fixed':
  for j in np.unique(q['lab_id']):
   panels=np.concatenate([g['ref_coupons'][g['lab_id']==j] for g in [tr,te]]); m=q['lab_id']==j
   v=(q['coupon_features_raw'][m]-np.median(panels.mean(0)))/np.median(panels.std(0))
   u[m]=np.where(np.isnan(v),x[m],v)
 for subset in [(),(0,1,2,3)]:
  A=np.column_stack((np.ones(len(x)),beta[2]+u@beta[3:],u[:,subset])); pinv=np.linalg.pinv(A); c=pinv@C[:,2]; e=C[:,2]-A@c; loo=e/(1-np.sum(A*pinv.T,axis=1)); print(variant,subset,'linear-drift',c.round(4),'rmse',np.mean(e**2)**.5,'loo',np.mean(loo**2)**.5)
A=np.column_stack((x, x*x, np.tanh(x),r,r*r,H,Z,P,np.diff(P,axis=1),P.mean(1),P.std(1),P.max(1),P.min(1),np.maximum(P,0)@W,np.minimum(P,0)@W)); names=[f'{name}{k}' for name,n in [('x',4),('sqx',4),('tanhx',4),('r',2),('h',1),('z',2),('past',6),('dp',5),('mean',1),('std',1),('max',1),('min',1),('positive_hist',2),('negative_hist',2)] for k in range(n)]
for j in range(3):
 corr=np.corrcoef(np.column_stack((A,C[:,j])).T)[-1,:-1]; print(j,[(names[k],round(corr[k],3)) for k in np.argsort(-abs(corr))[:12]])
print('diag missing positions',np.argwhere(np.isnan(q['coupon_features_raw'])).tolist()); print('stress ranges',[(float(np.min(v)),float(np.max(v))) for v in [P,H,stress(te['past_channel_raw']),stress(te['hold_channel_raw'])]])
