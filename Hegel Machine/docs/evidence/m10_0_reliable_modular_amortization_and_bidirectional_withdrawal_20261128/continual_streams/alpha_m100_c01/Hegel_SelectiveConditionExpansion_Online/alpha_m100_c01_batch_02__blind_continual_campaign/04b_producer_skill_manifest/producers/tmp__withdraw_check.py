import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
from chamber_probe_base import _features
from scipy.optimize import least_squares
xd=_features(q,q)[0]; good=np.isfinite(q['coupon_features_raw']).all(1); dr=augment(xd)@b[2:]; base=intrinsic(q,xd); resid=obs-base
print('times',t,'history dt',dt,'diagnostic lab counts',np.unique(q['lab_id'],return_counts=True))
def shape(tau): return np.r_[t+tau*np.expm1(-t/tau),0.]
def correction(v): return dr[:,None]*v[0]*shape(v[1])
def fit(mask):
 return least_squares(lambda v:(correction(v)-resid)[mask].ravel(),[-1,1],bounds=([-4,.1],[0,5]),loss='soft_l1',f_scale=.004).x
print('fit complete',fit(good),'fit all',fit(np.ones(len(good),bool)))
for tag,labels in [('coupon',np.arange(len(good))),('lab',q['lab_id']),('stratum',q['stratum'])]:
 e=np.empty_like(resid); fits=[]
 for group in np.unique(labels):
  held=labels==group; v=fit(good&~held); fits.append(v); e[held]=resid[held]-correction(v)[held]
 for label,sel in [('complete',good),('missing',~good)]:
  a=e[sel]; print(tag,label,'curve/rate rms',np.sqrt(np.mean(a[:,:-1]**2)),np.sqrt(np.mean(a[:,-1]**2)),'worst group curve/rate',np.max([np.r_[np.sqrt(np.mean(e[sel&(labels==g),:-1]**2)),np.sqrt(np.mean(e[sel&(labels==g),-1]**2))] for g in np.unique(labels[sel])],0))
 print(tag,'parameter ranges',np.min(fits,0),np.max(fits,0))
e=resid-correction([-1,1]); print('unit kernel complete per-horizon RMS',np.sqrt(np.mean(e[good]**2,0))); print('unit kernel missing per-case RMS',np.sqrt(np.mean(e[~good]**2,1))); print('rates observed/pred',np.c_[obs[:,-1],base[:,-1]])
