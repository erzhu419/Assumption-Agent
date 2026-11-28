import sys, json, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import _verified_upstream as up
import alpha_m100_c02_batch_04_solver as a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=p['dataset']; dg=p['campaign_diagnostics']; cal=up.calibrate_upstream(raw); d=up.calibrate_upstream({**raw,'test':dg})['test']
dt,t,_,tau,th=a._intrinsic(cal)
print('tau theta',tau,th)
tr,te=cal['train'],cal['test']; past,h=a._stress(d)
def joined(*blocks): return {k:np.concatenate([b[k] for b in blocks]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
def median(s): return np.median(np.abs(s))
for nm,x in [('tr+dg',a._features(tr,d)[-len(h):]),('all',a._features(joined(tr,te),d)[-len(h):]),('dg only',a._features(d,d)[-len(h):])]:
 z=np.zeros((len(h),2))
 for dur,s in zip(dt,past.T):
  q=np.exp(-dur/tau); z=q*z+(1-q)*s[:,None]
 rise=-np.expm1(-t[:,None]/tau); drift=x@th[2:]; y=d['strain_curves_obs']-d['strain_0'][:,None]; r=d['instant_rate_obs']; ht=h[:,None]*th[:2]; zt=z*th[:2]
 zero=np.zeros((len(h),len(t))); zeror=np.zeros(len(h)); Y=np.column_stack([y,r]); base=np.column_stack([ht@rise.T+drift[:,None]*t,np.sum(ht/tau,axis=1)+drift]); st=np.column_stack([zt@rise.T,np.sum(zt/tau,axis=1)])
 def col(v,k,r0):return np.column_stack([v[:,None]*k,v*r0])
 state=[col(-zt[:,j],rise[:,j],1/tau[j]) for j in range(2)]
 hold=[col(ht[:,j],rise[:,j],1/tau[j]) for j in range(2)]
 driftcols=[col(drift,t,1)]+[col(drift,rise[:,j]*tau[j]-t,0) for j in range(2)]
 cross=[col(-z[:,i]*th[j],rise[:,j],1/tau[j]) for i,j in ((0,1),(1,0))]
 driftfree=[col(x[:,i],t,1) for i in range(5)]+[col(x[:,i],rise[:,j]*tau[j]-t,0) for j in range(2) for i in range(5)]
 print(nm,'base(reset) med',median((Y-base)[:,:-1]),median((Y-base)[:,-1]),'base(history)',median((Y-base+st)[:,:-1]),median((Y-base+st)[:,-1]));
 w=np.r_[np.ones(len(t)),2.]; lab=d['lab_id']
 for name,cols in [('reset+drift',driftcols[1:]),('retain+drift',[sum(state)]+driftcols[1:]),('branch+drift',state+driftcols[1:]),('hold+branch+drift',hold+state+driftcols),('cross',hold+state+cross+driftcols),('driftfree',hold+state+driftfree)]:
  A=np.stack(cols,axis=-1)
  def fit(sel):
   B=(A[sel]*w[:,None]).reshape(-1,len(cols)); v=((Y-base)[sel]*w).ravel(); q=np.linalg.lstsq(B,v,rcond=None)[0]; return least_squares(lambda q:B@q-v,q,loss='soft_l1',f_scale=.006).x
  q=fit(np.ones(len(h),bool)); err=Y-base-A@q; cv=np.zeros_like(err)
  for j in np.unique(lab):
   s=lab==j; cv[s]=Y[s]-base[s]-A[s]@fit(~s)
  print(name,np.round(q,4),'in',np.round([median(err[:,:-1]),median(err[:,-1])],5),'LOLO typical/worst',np.round([median(cv[:,:-1]),median(cv[:,-1]),max(median(cv[lab==j,:-1]) for j in set(lab)),max(median(cv[lab==j,-1]) for j in set(lab))],5))
