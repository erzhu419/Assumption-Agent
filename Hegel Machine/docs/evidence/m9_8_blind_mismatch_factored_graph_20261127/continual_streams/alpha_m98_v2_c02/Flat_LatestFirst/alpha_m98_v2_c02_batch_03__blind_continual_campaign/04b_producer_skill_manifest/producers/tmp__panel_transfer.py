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
raw=cal['bridge_raw']; L=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],cal['true_channel_loads'],rcond=None)[0]; U=q['past_channel_raw']@L[:4]+L[4]; V=q['hold_channel_raw']@L[:4]+L[4]
keys=('lab_id','stratum','coupon_features_raw','ref_coupons')
join=lambda *groups:{k:np.concatenate([g[k] for g in groups]) for k in keys}
for name,panel in [('train/test+diag',join(tr,te)),('train+diag',tr),('test+diag',te),('diag only',q)]:
 xx=b._standardize(panel,q)[1]
 S1,R1=b._design(q,xx,stress,dt,t)(tau)
 rr=R1@beta-q['instant_rate_obs']; drift=R1[:,2:]@beta[2:]
 ee=S1@beta-drift[:,None]*G(1.)[:-1]+q['strain_0'][:,None]-q['strain_curves_obs']
 print(name,'strain/rate MAE',round(np.mean(np.abs(ee)),5),round(np.mean(np.abs(rr)),5),'rate RMSE',round(np.mean(rr**2)**.5,5),'max rate error',round(max(abs(rr)),5))
print('large residual coupons (row, lab, rate residual, past stresses)')
m=abs(yr)>.05
print(np.c_[np.arange(len(P)),q['lab_id'],yr,P][m].round(3))
Utest=te['hold_channel_raw']@L[:4]+L[4]
Pg=stress(te['past_channel_raw'])
for name,u in [('diagnostic',V[:,0]+P.std(1)),('deployment',Utest[:,0]+Pg.std(1))]:
 print(name,'history/channel index quantiles',np.quantile(u,[0,.1,.5,.9,1]).round(4),'below fitted hinge',int(sum(u<1.392)))
