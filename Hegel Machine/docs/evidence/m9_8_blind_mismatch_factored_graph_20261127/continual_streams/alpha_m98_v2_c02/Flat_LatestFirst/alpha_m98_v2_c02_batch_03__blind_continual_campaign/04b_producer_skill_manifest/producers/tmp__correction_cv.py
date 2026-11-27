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
from scipy.optimize import minimize_scalar
from itertools import combinations
n=len(P); ix=np.arange(n)
def moment(lag):
 return np.r_[-np.expm1(-t/lag)*lag,1.]
def pred(p,m):
 dur,lag=p[:2]; return -r[m,None]*G(dur)+p[2:,None]*moment(lag)
init=np.r_[1.,1.,yr]; opt=least_squares(lambda p:(pred(p,ix)-Y).ravel(),init,bounds=(np.r_[.3,.15,np.full(n,-2)],np.r_[2.,5.,np.full(n,2)])).x
print('offset transient duration, amp',opt.round(4),'joint rmse',np.sqrt(np.mean((pred(opt,ix)-Y)**2)))
A={ 'range_gate': np.ptp(P,axis=1)*V[:,0], 'std_gate':P.std(1)+V[:,0], 'channel_balance': V[:,0]-V[:,3], 'history_spread': P.std(1), 'drift':r }
for k in range(P.shape[1]):
 A['P'+str(k)]=P[:,k]; A['jump'+str(k)]=np.diff(np.c_[np.zeros(n),P],axis=1)[:,k]
for k in range(4):
 A['x'+str(k)]=x[:,k]; A['V'+str(k)]=V[:,k]
# Fit temporal law and coupon amplitude jointly. All thresholds and drift durations are fitted inside each outer fold.
base=np.column_stack((t,)); sig=moment(1.)
def fit(m,k=None,reg=.04):
 if k is None:
  dur=minimize_scalar(lambda d:np.mean((Y[m]+r[m,None]*G(d))**2),bounds=(.5,1.6),method='bounded').x
  return -r[:,None]*G(dur),[dur]
 u=A[k]; mu=u[m].mean(); sd=u[m].std(); z=(u-mu)/sd
 def fn(p):
  return -r[:,None]*G(p[0])+(p[1]-p[2]*np.minimum(np.maximum(p[3]-z,0),2.5))[:,None]*sig
 def err(p):return np.r_[(fn(p)[m]-Y[m]).ravel(),reg*p[2],.12*p[1]]
 p=min((least_squares(err,[1,0,.04,k0],bounds=([.6,-.1,0,-2.5],[1.4,.1,.22,1.5])) for k0 in (-1.8,-1,0)),key=lambda f:f.cost).x
 return fn(p),np.r_[p,mu,sd]
for k in (None,'range_gate','std_gate','channel_balance','history_spread','drift'):
 for group_name,group in [('coupon',ix),('lab',q['lab_id']),('stratum',q['stratum'])]:
  e=np.zeros_like(Y)
  for g in np.unique(group):
   m=group!=g; e[~m]=(fit(m,k)[0]-Y)[~m]
  print(k,group_name,'RMSE late rate',np.sqrt(np.mean(e[:,[-2,-1]]**2,0)).round(4),'max',np.max(abs(e[:,[-2,-1]]),0).round(4))
 print('full',np.asarray(fit(ix>=0,k)[1]).round(5))
