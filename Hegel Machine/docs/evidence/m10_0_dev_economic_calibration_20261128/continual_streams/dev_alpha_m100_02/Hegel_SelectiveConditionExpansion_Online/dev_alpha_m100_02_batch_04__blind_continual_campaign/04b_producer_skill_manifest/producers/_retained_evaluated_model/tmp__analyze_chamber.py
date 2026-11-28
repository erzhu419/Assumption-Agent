import pickle,numpy as np
from _verified_upstream import calibrate_upstream
from downstream_solver import _fit,_features,_response,_design
z=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=calibrate_upstream(dict(z['dataset'],chamber_verification_trials=z['campaign_diagnostics'])); tr=d['train']; te=d['test']; dg=d['chamber_verification_trials']; dt=d['past_dt']; t=d['eval_times']; x,xe,xd=_features(tr,te,dg); tau,th=_fit(tr,x,dt,t); print('TAU,TH',tau,th)
for s,xx in [(tr,x),(dg,xd)]:
 a=_design(s,xx,dt,t,tau); y=_response(s); e=y-a@th; print('residual abs',np.quantile(abs(e),[.25,.5,.9,.99],axis=0).round(4)); print('residual Mean',e.mean(axis=0).round(4)); print('by coupon first,last,rate',e[:,[0,-2,-1]].round(3)); print('OLS',np.linalg.lstsq(a.reshape(-1,7),y.ravel(),rcond=None)[0])
