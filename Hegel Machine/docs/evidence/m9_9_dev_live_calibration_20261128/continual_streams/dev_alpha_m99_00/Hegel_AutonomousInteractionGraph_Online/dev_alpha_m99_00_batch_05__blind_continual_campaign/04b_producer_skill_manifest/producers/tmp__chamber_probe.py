import sys,json,pickle
sys.path.insert(0,'/holdout_task')
sys.path.insert(0,'/tmp/hegel_continual_archive')
import numpy as np
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
import dev_alpha_m99_00_batch_02_solver as old
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; q=p['campaign_diagnostics']; d=calibrate_upstream({**d,'chamber_verification_trials':q.get('chamber_verification_trials',q)})
a,b,c=(d[k] for k in ('train','test','chamber_verification_trials')); t=d['eval_times']; dt=d['past_dt']; n=len(a['lab_id']); m=len(b['lab_id'])
x=old._features(a,b,c); cp=old._coupon(c,x[n+m:]); taus,theta=old._fit(old._coupon(a,old._features(a,b)[:n]),dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']); print('material',taus,theta)
y=c['strain_curves_obs']-c['strain_0'][:,None]; rate=c['instant_rate_obs']; past=c['past_channel_raw'].sum(-1); hold=c['hold_channel_raw'].sum(-1); drift=theta[2]+x[n+m:]@theta[3:]; Y=np.c_[y,rate]; print('range past',np.round(np.percentile(past,[0,50,100],axis=0),3),'hold',np.percentile(hold,[0,50,100]));
def basis(T): return np.c_[np.vstack((-np.expm1(-t[:,None]/T),1/T)),np.r_[t,1.]]
def profile(logT, ret=False):
 B=basis(np.exp(logT)); W=np.linalg.lstsq(B,Y.T,rcond=None)[0].T; R=W@B.T-Y
 return (W,R) if ret else R.ravel()
for k in (1,2,3,4):
 fits=[least_squares(profile,np.log(s),bounds=(np.log(.035),np.log(25)),max_nfev=250) for s in [np.geomspace(.25,4,k),np.geomspace(.5,12,k),np.geomspace(.08,2,k)]]; fit=min(fits,key=lambda f:f.cost); T=np.sort(np.exp(fit.x)); W,R=profile(np.log(T),True); print('free branches',T,'relative',T[:,None]/taus,'rmse',np.sqrt(np.mean(R**2)),'rate',np.sqrt(np.mean(R[:,-1]**2)),'max',np.max(abs(R)))
 if k==2 or k==3:
  S=np.c_[hold,drift,past,x[n+m:]]; print('coefficient range',np.round(np.percentile(W,[0,50,100],axis=0),3));print('amplitude corr hold, drift, past[6], x[4]',np.round(np.corrcoef(W.T,S.T)[:k+1,k+1:],2))
