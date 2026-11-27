import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import *
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds = calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,d = (ds[k] for k in ('train','test','chamber_verification_trials'))
dt,t=ds['past_dt'],ds['eval_times']
x,xt,xd=features(tr,te,d)
taus,theta=material(tr,x,dt,t)
print('dt,tau,theta',dt,taus,theta)
for split, xx in [(tr,x),(d,xd)]:
 C,R=basis(taus,*loads(split),xx,dt,t)
 dy=split['strain_curves_obs']-split['strain_0'][:,None]-C@theta
 dr=split['instant_rate_obs']-R@theta
 print('curve error quantiles by time',np.round(np.quantile(dy,[0,.25,.5,.75,1],axis=0),4))
 print('rate errors',np.round(dr,4) if split is d else np.quantile(dr,[0,.25,.5,.75,1]))
 if split is d:
  drift=theta[2]+xx@theta[3:]
  print('drift and curve errors (begin,middle,end)',np.round(np.c_[drift,dy[:,[0,5,-1]],dr],4))
