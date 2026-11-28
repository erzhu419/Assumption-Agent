from explore import *
from scipy.optimize import least_squares
p,h,F=inputs[2]; Y=di['strain_curves_obs']-di['strain_0'][:,None]; R=di['instant_rate_obs']; yy=np.c_[Y,R]
def basis(tau): return np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
def inspect(A, free_drift):
 D=basis(tau)[:,:2+free_drift]; gains=np.linalg.lstsq(D,yy.T,rcond=None)[0].T; pred=np.empty_like(yy)
 for lab in np.unique(di['lab_id']):
  msk=di['lab_id']!=lab; pred[~msk]=A[~msk]@np.linalg.lstsq(A[msk],gains[msk],rcond=None)[0]@D.T
 print('  CV curve/rate RMS', rms(pred[:,:-1]-Y),rms(pred[:,-1]-R),'max', np.max(abs(pred[:,:-1]-Y)),np.max(abs(pred[:,-1]-R)))
 print('  coefs',np.round(np.linalg.lstsq(A,gains,rcond=None)[0].T,4)); print('  independent RMS',rms(gains@D.T-yy))
def linked(start, drift):
 def err(logtau):
  D=basis(np.exp(logtau))[:,:2+drift]; return (yy-np.linalg.lstsq(D,yy.T,rcond=None)[0].T@D.T).ravel()
 fit=least_squares(err,np.log(start)); print('chamber free taus',drift,np.exp(fit.x),rms(fit.fun)); print('fixed',rms(err(np.log(tau))))
for drift in [0,1]:
 linked(tau,drift)
 for name,A in [('hold',np.c_[np.ones(len(h)),h]),('hold past',np.c_[np.ones(len(h)),h,p]),('hold past features',np.c_[h,p,F])]:
  print(drift,name); inspect(A,drift)
print('row detail, hold past X amplitudes'); D=basis(tau)[:,:2]; print(np.round(np.c_[h,p,F[:,1:],np.linalg.lstsq(D,yy.T,rcond=None)[0].T],3))
for s in (tr,te,di):
 raw=s['hold_channel_raw']; pp=s['past_channel_raw']; print('raw ranges',np.round(raw.min(0),3),np.round(raw.max(0),3),'prev',np.round(pp.min((0,1)),3),np.round(pp.max((0,1)),3))
